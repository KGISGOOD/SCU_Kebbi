"""Native-template assistant-only labels; fail rather than silently truncate."""
from .common import DataError


def encode_sample(tokenizer, row, max_length):
    messages = row["messages"]
    # Native template includes all role headers and answer end-of-turn tokens.
    prefix_text = tokenizer.apply_chat_template(messages[:2], tokenize=False,
                                                add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)
    if not full_text.startswith(prefix_text):
        raise DataError("%s: chat template has no exact assistant prefix boundary" % row["faq_id"])
    prefix = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise DataError("%s: token boundary mismatch/empty answer; refusing guessed labels" % row["faq_id"])
    answer_ids = tokenizer(messages[2]["content"], add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise DataError("%s: no answer tokens" % row["faq_id"])
    if len(full) > max_length:
        raise DataError("%s: %d tokens exceeds max length %d; raise --max-length, no truncation"
                        % (row["faq_id"], len(full), max_length))
    return {"faq_id": row["faq_id"], "input_ids": full,
            "labels": [-100] * len(prefix) + full[len(prefix):],
            "supervised_tokens": len(full) - len(prefix)}


def collate(samples, pad_id, device):
    import torch
    length = max(len(row["input_ids"]) for row in samples)
    ids, masks, labels = [], [], []
    for row in samples:
        padding = length - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * padding)
        masks.append([1] * len(row["input_ids"]) + [0] * padding)
        labels.append(row["labels"] + [-100] * padding)
    # Right padding is masked by POSITION, not token ID (pad may equal EOS).
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device)}
