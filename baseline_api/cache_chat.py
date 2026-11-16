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
    print('模式:', engine.mode, '自動暫存:', config['cache'].get('auto_write', False))
    print('輸入 exit 結束；approve-last 為人工確認；Experimental暫存標為unverified，不代表答案正確。')
    print('快取僅本次程序有效；啟動完整Hash＋Windows唯讀保護，查詢不重算大型權重Hash。')
    if config['cache']['verify_sources_each_request'] or (engine.guard is not None and not getattr(engine.guard, 'locked', False)):
        print('目前使用完整Hash逐次驗證（舊設定或平台不支援唯讀保護）；可能較慢。')
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
                print('問題:', question)
                print('來源:', detail['answer_source'], 'Cache:', 'Hit' if detail['cache_hit'] else 'Miss',
                      '相似度:', detail['similarity'], '原因:', detail['reason'])
                print('命中類型:', detail['hit_type'], '答案驗證狀態:', detail['answer_verification'])
                print('耗時: %.3f 秒；字數: %d；超過100字: %s' % (elapsed, len(answer), len(answer) > 100))
                print('快取筆數:', detail['cache_size'], '本次自動寫入:', detail['saved'])
                print('寫入狀態:', detail['save_status'])
                print('階段耗時（秒）:', detail['timings'])
                for doc in (detail.get('rag_detail') or {}).get('documents', []):
                    print('檢索來源:', doc['metadata'], '\n', doc['content'])
            except Exception as exc:
                print('失敗:', exc)
        except (EOFError, KeyboardInterrupt):
            break
    engine.close()


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
