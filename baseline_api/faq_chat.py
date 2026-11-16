"""Single-turn Baseline 2 Experimental terminal client; no HTTP server."""
import argparse
import json
from pathlib import Path

from .faq import FAQFallback, validate_faq_config

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).expanduser()
    path = path if path.is_absolute() else ROOT / path
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    validate_faq_config(config.get('faq'))
    if (config.get('prompt_version') != 'baseline_senior_v2'
            or config['faq'].get('acceptance_mode') != 'experimental'
            or config['faq']['threshold'] != 0.9 or config['faq'].get('min_margin') != 0.05):
        raise ValueError('requires baseline_senior_v2 / experimental / threshold=0.9 / min_margin=0.05')
    for name in ('model', 'bge', 'knowledge_index'):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError('missing local resource path: ' + name)
    return config


def chat(config_path):
    from experiments.shared.a0_adapter import A0Adapter, measured
    config = load_config(config_path)
    print('載入 Baseline 2：LLM、BGE-M3、Dense Index 及 FAQ 索引……')
    adapter = A0Adapter(ROOT, config)
    engine = FAQFallback(ROOT, config, adapter)
    if (engine.pipeline.prompt_version != 'baseline_senior_v2'
            or engine.identity['prompt_version'] != 'baseline_senior_v2'
            or engine.acceptance_mode != 'experimental'):
        raise ValueError('loaded Prompt or acceptance mode differs from requested configuration')
    print('Prompt: baseline_senior_v2；FAQ: experimental；threshold=0.9；min_margin=0.05')
    print('門檻尚未校準，Experimental 回答不代表人工驗證；單輪問答，輸入 exit 結束。')
    while True:
        try:
            question = input('您的問題: ').strip()
            if question.lower() in ('exit', 'quit', 'q'):
                break
            if not question:
                continue
            try:
                (answer, detail), elapsed = measured(engine.answer, question)
                print('回答:', answer)
                print('來源:', detail['answer_source'])
                print('FAQ Top-1:', detail['faq_top1_score'], 'Top-2:', detail['faq_top2_score'],
                      '分數差距:', detail['faq_score_margin'])
                print('FAQ接受:', detail['faq_accepted'], '原因:',
                      detail['faq_acceptance_reason'] or detail['faq_rejection_reason'])
                print('耗時: %.3f 秒；字數（含標點及空白）: %d；超過100字: %s' %
                      (elapsed, len(answer), '是' if len(answer) > 100 else '否'))
            except Exception as exc:
                print('問答失敗:', exc)
        except (EOFError, KeyboardInterrupt):
            break
    print('再見！')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', default='experiments/configs/a0_faq_baseline_v2.experimental.json')
    args = parser.parse_args()
    try:
        chat(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, 'Baseline 2 啟動失敗: %s\n' % exc)


if __name__ == '__main__':
    main()
