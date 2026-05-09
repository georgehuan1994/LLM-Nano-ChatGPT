"""
这个脚本负责统一评估 base model。

支持三种评估模式，可以组合使用：
1. `core`   : 在 CORE 基准上评估任务能力
2. `bpb`    : 在 train/val 数据上评估 bits per byte
3. `sample` : 让模型生成若干样例文本，做人工直观检查

默认三种都跑：

    --eval core,bpb,sample

对初学者来说，可以把它理解成：

- `base_train.py` 负责 “让模型学”
- `base_eval.py` 负责 “检查模型学得怎么样”

其中：
- BPB 更像 “语言建模内功分”
- CORE 更像 “做题能力分”
- sample 更像 “现场口头表现”
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
import argparse
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# HuggingFace 模型适配工具
#
# 这个脚本既能评估 nanochat 自己训练的模型，
# 也能评估 HuggingFace 上现成的模型。
#
# 但两者接口不完全一样，所以这里做一个轻量适配层。

class ModelWrapper:
    """
    给 HuggingFace 模型套一个 nanochat 风格接口。

    这样后面无论是 nanochat 自己的 GPT，还是外部 HF 模型，
    评估代码都可以统一地写成：

        model(x, y, loss_reduction=...)

    """
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        # HuggingFace 因果语言模型通常返回 logits。
        # logits 可以理解成：
        # “在当前位置，模型对词表里每个 token 的打分”
        logits = self.model(input_ids).logits
        if targets is None:
            return logits

        # 如果给了 targets，就直接在包装层里算交叉熵 loss，
        # 这样外部看起来就和 nanochat 的 GPT.forward(...) 行为一致。
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """加载 HuggingFace 模型和 tokenizer，并包装成项目统一接口。"""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """
    为 HuggingFace tokenizer 动态构造 token_bytes。

    nanochat 自己训练的 tokenizer 会在 tok_train.py 里直接把 token_bytes.pt 落盘。
    但外部 HF tokenizer 通常没有这份配套文件，所以这里现场计算一份。

    核心含义仍然一样：
        token_id -> 该 token 对应多少个 UTF-8 字节
    """
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes

# -----------------------------------------------------------------------------
# CORE 评估
#
# CORE 可以理解成一个 “任务能力基准”。
# 它不只是问：
#   “模型会不会预测下一个 token”
# 而是更关心：
#   “模型在一批 few-shot / in-context learning 任务上表现如何”

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """把下载下来的 eval bundle 解压到项目 base_dir 下。"""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """
    在 CORE benchmark 上评估模型。

    返回值包含三部分：
    - results:            每个任务的原始准确率
    - centered_results:   以随机基线为参照做过中心化后的分数
    - core_metric:        所有 centered_results 的平均值

    对初学者来说，一个比较好懂的理解方式是：
    CORE 不是在测“语言模型 loss 漂不漂亮”，
    而是在测“这个模型拿去做具体 few-shot 任务时，有没有表现出泛化能力”。
    """
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # 如果本地还没有评测包，就先下载。
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # 读取每个任务的随机基线。
    # 后面会用它来做 centered score，避免不同任务直接看原始准确率不太可比。
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # 逐个任务评估
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # 如果开启 max_per_task，为了让抽样更稳定、可复现，先固定随机种子打乱。
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # evaluate_task 会真正把 few-shot 提示构造出来，送入模型做任务预测。
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]

        # centered_result 的作用可以粗略理解成：
        # “把随机乱猜的分数平移掉，再缩放到更可比的区间里”
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# 主流程

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    # 解析用户要求运行哪些评估模式。
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # 初始化设备 / 分布式环境。
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # 加载模型与 tokenizer。
    #
    # 分两条分支：
    # 1. 如果给了 --hf-path，就评估 HuggingFace 模型
    # 2. 否则评估 nanochat 自己训练保存的 checkpoint
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)
        model_name = f"base_model (step {meta['step']})"
        model_slug = f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # 下面这些变量用于汇总不同评估分支的结果，最后统一写报告。
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    #
    # 采样不是严格数值指标，但它非常适合人类快速感知模型：
    # - 是否会接着提示往下说
    # - 是否学到了一些事实和模式
    # - 输出是否明显崩坏
    if 'sample' in eval_modes and not is_hf_model:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            # Engine 可以理解成一个“推理驱动器”：
            # 它负责反复调用模型，按自回归方式一个 token 一个 token 地往后生成。
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                # prepend BOS 的作用是明确告诉模型 “这是一个新序列的开始”。
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    elif 'sample' in eval_modes and is_hf_model:
        print0("\nSkipping sampling for HuggingFace models (not supported)")

    # --- BPB evaluation ---
    #
    # 这部分最接近 “语言模型本职工作” 的评估：
    # 在真实 train/val token 流上，模型预测下一个 token 的能力如何。
    #
    # 但项目不是简单输出 mean loss，而是输出 BPB，
    # 这样在 tokenizer 粒度变化时更公平。
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        # 每个评估 step 实际会处理多少 token。
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # 让 split_tokens 向下对齐到 tokens_per_step 的整数倍，
            # 否则没法整齐地跑完整个 step 数。
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            # 这里的 loader 会把文本编码成 token 序列，并整理成语言模型评估需要的 x/y 批次。
            loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            bpb = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    #
    # 如果说 BPB 更像 “基本功分数”，
    # 那 CORE 更像 “拿模型去做实际 few-shot 任务时的表现分数”。
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)

        # 把每个 CORE 子任务的结果写成 CSV，方便后续做表格分析。
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")

    # --- Log to report ---
    #
    # 把不同评估分支的结果整理后统一写入 report 系统，
    # 便于后续查看实验记录。
    from nanochat.report import get_report
    report_data = [{"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data.append(core_results["centered_results"])

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    get_report().log(section="Base model evaluation", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
