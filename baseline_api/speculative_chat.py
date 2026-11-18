"""V6 terminal client; shared routing and Target-verified generation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from .server import load_settings, ROOT
from .speculative import validate_speculative
from .timing_output import seconds_text, timing_json, executed_stage_timings


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', help='NEW child of experiments/results')
    args = parser.parse_args()
    config = load_settings(args.config)
    validate_speculative(config)
    output = Path(args.run_dir) if args.run_dir else Path('experiments/results/speculative_cli') / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    output = (output if output.is_absolute() else ROOT / output).resolve()
    if (ROOT / 'experiments/results').resolve() not in output.parents:
        parser.error('run-dir must be a NEW child of experiments/results')
    output.mkdir(parents=True, exist_ok=False)
    engine = None
    try:
        from .integrated_chat import IntegratedQA
        engine = IntegratedQA(ROOT, config)
        with (output / 'session.json').open('x', encoding='utf-8') as stream:
            json.dump({'config': config, 'identity': engine.identity,
                       'formal_quality_certified': False}, stream, ensure_ascii=False, indent=2)
        print('Prompt:', config['prompt_version'], 'Greedy SD; Target:', config['model'],
              'Draft:', config['speculative']['draft_model'])
        print('逐題 Greedy 對照:', config['speculative']['verify_greedy'],
              '（開啟時總耗時包含額外對照生成；未保證真實模型 Token 等價）')
        print('輸入 exit 結束。紀錄:', output)
        with (output / 'results.jsonl').open('x', encoding='utf-8') as stream:
            while True:
                try:
                    question = input('問題: ').strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if question.lower() in ('exit', 'quit', 'q'):
                    break
                if not question:
                    continue
                start = time.perf_counter()
                try:
                    answer, detail = engine.answer(question)
                    record = dict(status='success', question=question, answer=answer,
                                  answer_char_count=len(answer), exceeds_100_characters=len(answer)>100,
                                  detail=detail, total_seconds=time.perf_counter()-start)
                    sd = detail['speculative']
                    print('回答:', answer)
                    print('來源:', detail.get('answer_source', 'rag'), '字數:', len(answer),
                          '超過100字:', len(answer)>100, '總秒數:', seconds_text(record['total_seconds']))
                    print('SD:', timing_json({k: v for k, v in sd.items() if k not in
                                  ('raw_generation', 'paired_greedy', 'output_token_ids')}))
                    print('階段秒數:', timing_json(executed_stage_timings(detail)))
                    cache = detail.get('cache') or {}
                    rag = cache.get('rag_detail') or detail
                    if sd['used']:
                        print('RAG 檢索秒數:', seconds_text(rag.get('retrieval_seconds')),
                              'SD 生成秒數:', seconds_text(sd['generation_seconds']),
                              '額外 Greedy 驗證秒數:', seconds_text(sd['greedy_verification_seconds']))
                    record['detail'] = dict(detail, timings=executed_stage_timings(detail))
                except Exception as exc:
                    record = dict(status='failed', question=question, error=str(exc),
                                  speculative_failure=getattr(exc, 'speculative_failure', None),
                                  total_seconds=time.perf_counter()-start)
                    print('失敗，未回傳或快取答案:', exc)
                stream.write(timing_json(record)+'\n')
                stream.flush()
    finally:
        if engine is not None:
            engine.close()


if __name__ == '__main__':
    main()
