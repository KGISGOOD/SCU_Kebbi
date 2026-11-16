"""Independent human decisions bound to immutable FAQ candidates; no model imports."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

FIELDS = ['faq_id', 'question', 'answer', 'sources', 'original_review_status',
          'candidate_sha256', 'decision', 'reviewer', 'reviewed_at', 'evidence']


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
                               sources=[], review_status='needs_review', source_kind='two_column_csv',
                               dataset_sha256=dataset_hash))
    if len(candidates) != 138:
        raise ValueError('expected 138 FAQ CSV rows')
    return candidates, dataset_hash


def display(row):
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return dict(faq_id=row['intent_group_id'], question=row['question'], answer=row['answer'],
                sources=json.dumps(row['sources'], ensure_ascii=False),
                original_review_status=row['review_status'],
                candidate_sha256=hashlib.sha256(encoded.encode('utf-8')).hexdigest())


def read_reviews(path, candidates):
    """Fail closed on edited candidates, duplicate IDs or incomplete approval."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError('FAQ review CSV columns changed')
        rows = list(reader)
    originals = {r['intent_group_id']: r for r in candidates}
    decisions = {}
    for row in rows:
        key = row['faq_id']
        if key not in originals or key in decisions:
            raise ValueError('unknown/duplicate FAQ review ID: ' + key)
        original = originals[key]
        editable_sources = original.get('source_kind') == 'two_column_csv'
        if any(row.get(k) != v for k, v in display(original).items() if not (editable_sources and k == 'sources')):
            raise ValueError('FAQ review candidate changed: ' + key)
        try:
            sources = json.loads(row['sources'])
            if not isinstance(sources, list) or any(not isinstance(s, dict) for s in sources):
                raise ValueError('sources must be a JSON array of objects')
        except (ValueError, TypeError) as exc:
            raise ValueError('invalid review sources: ' + key) from exc
        decision = row.get('decision')
        if decision not in ('needs_review', 'approved', 'rejected'):
            raise ValueError('invalid FAQ decision: ' + key)
        if decision in ('approved', 'rejected'):
            if any(not (row.get(k) or '').strip() for k in ('reviewer', 'reviewed_at', 'evidence')):
                raise ValueError('reviewer, reviewed_at and evidence required: ' + key)
            try:
                when = datetime.fromisoformat(row['reviewed_at'].replace('Z', '+00:00'))
                if when.tzinfo is None:
                    raise ValueError('timezone required')
            except ValueError as exc:
                raise ValueError('reviewed_at must be ISO8601 with timezone: ' + key) from exc
        if decision == 'approved' and (not original['answer'].strip() or not sources
                or any(not isinstance(s.get('path'), str) or not s['path'].strip()
                       or not isinstance(s.get('quote'), str) or not s['quote'].strip() for s in sources)):
            raise ValueError('cannot approve empty answer or insufficient evidence: ' + key)
        decisions[key] = row
    if set(decisions) != set(originals):
        raise ValueError('FAQ review must retain every canonical ID')
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--review-file', type=Path, default=Path('experiments/datasets/faq_csv_review.csv'))
    parser.add_argument('--faq-file', type=Path, default=Path('experiments/datasets/faq_精簡問答_138筆.csv'))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.project_root.resolve()
    candidates, _ = load_csv_candidates(args.faq_file if args.faq_file.is_absolute() else root / args.faq_file)
    path = args.review_file if args.review_file.is_absolute() else root / args.review_file
    if not args.check_only:
        with path.open('x', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for row in candidates:
                writer.writerow(dict(display(row), decision='needs_review', reviewer='', reviewed_at='', evidence=''))
    decisions = read_reviews(path, candidates)
    print(json.dumps({'file': str(path), 'count': len(decisions),
                      'approved': sum(r['decision'] == 'approved' for r in decisions.values())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
