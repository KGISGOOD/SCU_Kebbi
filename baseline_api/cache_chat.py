"""Baseline 3 memory-only, reviewed-answer CLI; no HTTP/FAQ/SD."""
import argparse
import json
from pathlib import Path

from .cache import ReviewedCacheA0, validate_options

ROOT = Path(__file__).resolve().parents[1]


def chat(path):
    from experiments.shared.a0_adapter import A0Adapter, measured
    path = Path(path).expanduser()
    config = json.loads((path if path.is_absolute() else ROOT / path).read_text(encoding='utf-8-sig'))
    validate_options(config.get('cache'))
    if config.get('prompt_version') != 'baseline_senior_v2' or 'faq' in config:
        raise ValueError('requires Baseline 3 / baseline_senior_v2 without FAQ')
    engine = ReviewedCacheA0(A0Adapter(ROOT, config), config['cache'])
    print('Baseline 3；baseline_senior_v2；Cache門檻尚未校準；單輪問答。')
    print('輸入 exit 結束；approve-last 可人工核對並保存上一則RAG回答；未批准回答不自動保存。')
    print('每次請求驗證模型、索引及程式Hash；此成本計入耗時。快取僅本次程序有效。')
    while True:
        try:
            question = input('您的問題: ').strip()
            if question.lower() in ('exit', 'quit', 'q'):
                break
            if not question:
                continue
            try:
                if question == 'approve-last':
                    confirmed = input('已核對原文、年度資格、數字及回答完整性？確認請輸入 YES: ') == 'YES'
                    if not confirmed:
                        print('未批准。')
                        continue
                    reviewer = input('實際審核者: ')
                    evidence = input('來源位置及核對理由: ')
                    print('已保存:', engine.approve_last(reviewer, evidence, confirmed=True))
                    continue
                (answer, detail), elapsed = measured(engine.answer, question)
                print('回答:', answer)
                print('來源:', detail['answer_source'], 'Cache:', 'Hit' if detail['cache_hit'] else 'Miss',
                      '相似度:', detail['similarity'], '原因:', detail['reason'])
                print('耗時: %.3f 秒；字數: %d；超過100字: %s' % (elapsed, len(answer), len(answer) > 100))
                print('快取筆數:', detail['cache_size'], '本次自動寫入:', detail['saved'])
                for doc in (detail.get('rag_detail') or {}).get('documents', []):
                    print('檢索來源:', doc['metadata'], '\n', doc['content'])
            except Exception as exc:
                print('失敗:', exc)
        except (EOFError, KeyboardInterrupt):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', default='experiments/configs/a0_cache_baseline_v2.json')
    args = parser.parse_args()
    try:
        chat(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, 'Baseline 3 啟動失敗: %s\n' % exc)


if __name__ == '__main__':
    main()
