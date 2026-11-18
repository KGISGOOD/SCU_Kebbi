"""Read-only two-column FAQ loader; no external review or ID registry."""
import csv
import hashlib
import json
from pathlib import Path


def load_csv_candidates(path):
    path = Path(path)
    raw = path.read_bytes()
    dataset_hash = hashlib.sha256(raw).hexdigest()
    import io
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != ['問題', '答案']:
        raise ValueError('FAQ CSV must contain exactly 問題,答案')
    candidates, seen = [], set()
    for row in reader:
        if None in row or not isinstance(row.get('問題'), str) or not row['問題'].strip() or row.get('答案') is None:
            raise ValueError('malformed FAQ CSV row')
        question, answer = row['問題'], row['答案']
        if question.strip() in seen:
            raise ValueError('duplicate FAQ question')
        seen.add(question.strip())
        key = 'FAQ_' + hashlib.sha256(json.dumps([question, answer], ensure_ascii=False).encode('utf-8')).hexdigest()
        candidates.append(dict(intent_group_id=key, question_id=key, question=question, answer=answer,
                               sources=[], source_kind='two_column_csv',
                               dataset_sha256=dataset_hash))
    if len(candidates) != 138:
        raise ValueError('expected 138 FAQ CSV rows')
    return candidates, dataset_hash
