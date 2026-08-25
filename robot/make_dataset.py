import json
import pathlib
import random

INSTRUCTION = "請回答使用者關於東吳大學資料科學系的問題。"

def main():
    # Path to your cleaned TSV file (question<TAB>answer)
    tsv_path = pathlib.Path("faq.txt")
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Cannot find {tsv_path.resolve()}")

    # Read all non‑empty lines
    lines = [ln.strip() for ln in tsv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    pairs = []
    for line in lines:
        # Split on the first tab only – answer may contain tabs if ever needed
        if "\t" not in line:
            # Skip malformed lines
            continue
        q, a = line.split("\t", 1)
        pairs.append((q.strip(), a.strip()))

    if not pairs:
        raise ValueError("No valid Q&A pairs found in faq.txt")

    print(f"Loaded {len(pairs)} Q&A pairs from {tsv_path}")

    # Shuffle with a fixed seed for reproducibility
    random.seed(42)
    random.shuffle(pairs)

    # 90% train, 10% validation
    split_idx = int(0.9 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    def write_jsonl(pairs_list, out_path):
        with out_path.open("w", encoding="utf-8") as f:
            for q, a in pairs_list:
                obj = {
                    "instruction": INSTRUCTION,
                    "input": q,
                    "output": a
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    train_path = pathlib.Path("faq_train.jsonl")
    val_path   = pathlib.Path("faq_val.jsonl")

    write_jsonl(train_pairs, train_path)
    write_jsonl(val_pairs,   val_path)

    print(f"Written {len(train_pairs)} training samples to {train_path}")
    print(f"Written {len(val_pairs)}   validation samples to {val_path}")

if __name__ == "__main__":
    main()