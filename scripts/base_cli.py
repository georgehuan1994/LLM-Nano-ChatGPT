"""
Simple base-model command-line completion.

This is intentionally separate from chat_cli.py: base checkpoints are plain language
models, not instruction/chat models, so we feed a raw text prompt and stream the
continuation back.
"""
import argparse
import codecs
import torch

from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model


parser = argparse.ArgumentParser(description="Complete text with a nanochat base model")
parser.add_argument("-i", "--source", type=str, default="base", choices=["base"], help="Source of the model: base")
parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
parser.add_argument("-p", "--prompt", type=str, default="", help="Prompt to complete once; empty starts interactive mode")
parser.add_argument("-m", "--max-tokens", type=int, default=128, help="Maximum number of tokens to generate")
parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Temperature for generation")
parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling parameter")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type: cuda|cpu|mps. empty => autodetect")
args = parser.parse_args()


def decode_stream_token(tokenizer, token, utf8_decoder):
    # base 模型也使用 byte-level BPE tokenizer。中文、emoji 等 UTF-8
    # 多字节字符可能跨 token；逐 token decode 会把半截字节打印成 "�"。
    # 这里复用 chat_cli 的修复：按 token bytes 做 UTF-8 增量解码。
    if hasattr(tokenizer, "enc") and hasattr(tokenizer.enc, "decode_single_token_bytes"):
        try:
            token_bytes = tokenizer.enc.decode_single_token_bytes(token)
        except KeyError:
            return tokenizer.decode([token])
        return utf8_decoder.decode(token_bytes, final=False)
    return tokenizer.decode([token])


device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, _, _, _, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)

print("\nNanoChat Base Completion Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end")
print("Each input is treated as a fresh base-model prompt")
print("-" * 50)

while True:
    if args.prompt:
        prompt = args.prompt
    else:
        try:
            prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if prompt.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    if not prompt:
        continue

    prompt_tokens = tokenizer(prompt, prepend="<|bos|>")
    utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    if args.prompt:
        print(f"\nPrompt: {prompt}")
    print("Completion: ", end="", flush=True)
    with torch.inference_mode():
        for token_column, token_masks in engine.generate(
            prompt_tokens,
            num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            token = token_column[0]
            print(decode_stream_token(tokenizer, token, utf8_decoder), end="", flush=True)

    tail_text = utf8_decoder.decode(b"", final=True)
    if tail_text:
        print(tail_text, end="", flush=True)
    print()

    if args.prompt:
        break
