"""Single-turn generation from base+LoRA; never opens FAQ answers or indexes."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from .common import BASE, ROOT, DataError, resolve
from .model_loader import load_adapter_for_chat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--base-model", default=BASE)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true", help="default: greedy generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-unreviewed", action="store_true")
    args = parser.parse_args()
    if (args.max_input_tokens < 1 or args.max_new_tokens < 1 or not 0 <= args.seed < 2**32
            or not math.isfinite(args.temperature) or args.temperature <= 0
            or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1):
        parser.error("invalid token limits, seed, temperature or top-p")
    root = args.project_root.expanduser().resolve()
    try:
        model, tokenizer, provenance = load_adapter_for_chat(
            root, args.base_model, resolve(root, args.adapter), args.device, args.precision,
            args.offline, args.allow_unreviewed)
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        parser.exit(2, "載入失敗 [%s]：%s\n" % (type(exc).__name__, exc))
    print("A3 FAQ SFT — 單輪 LLM 生成（無檢索）\nBase：%s\nAdapter：%s\nprovisional=%s" % (
        args.base_model, resolve(root, args.adapter), provenance["provisional"]))
    print("輸入 exit / quit 結束；生成回答可能有錯誤。")
    while True:
        try:
            query = input("\n問題：")
            if query.strip().lower() in ("exit", "quit"):
                return 0
            if not query.strip():
                print("請輸入非空白問題。")
                continue
            messages = [{"role": "system", "content": provenance["system_prompt"]},
                        {"role": "user", "content": query}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            length = inputs["input_ids"].shape[1]
            if length > args.max_input_tokens or length + args.max_new_tokens > model.config.max_position_embeddings:
                raise DataError("問題超過 token 限制，請縮短；不會默默截斷")
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            settings = {"max_new_tokens": args.max_new_tokens, "do_sample": args.do_sample,
                        "pad_token_id": tokenizer.pad_token_id}
            if args.do_sample:
                settings.update(temperature=args.temperature, top_p=args.top_p)
            # Keep base generation_config EOS/end-of-turn IDs; no hardcoded special tokens.
            with torch.inference_mode():
                output = model.generate(**inputs, **settings)
            answer = tokenizer.decode(output[0, length:], skip_special_tokens=True).strip()
            print("回答：\n%s" % (answer or "模型未生成可顯示內容。"))
        except (EOFError, KeyboardInterrupt):
            print("\n結束問答。")
            return 0
        except Exception as exc:
            print("無法生成 [%s]：%s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    raise SystemExit(main())
