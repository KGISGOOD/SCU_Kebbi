"""FAQ -> reviewed/experimental cache -> existing A0; single-process CLI only."""
import argparse
import json
from pathlib import Path
import time
from datetime import datetime, timezone
import uuid

from .faq import FAQFallback, validate_faq_config
from .cache import ReviewedCacheA0, ResourceGuard, validate_options
from experiments.shared.a0_adapter import A0Adapter, measured, file_hash

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).expanduser()
    path = path if path.is_absolute() else ROOT / path
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    validate_faq_config(config.get('faq'))
    validate_options(config.get('cache'))
    if (config.get('prompt_version') != 'baseline_senior_v2'
            or config['faq'].get('acceptance_mode') != 'experimental'
            or config['faq']['threshold'] != 0.9 or config['faq'].get('min_margin') != 0.05
            or config['cache'].get('acceptance_mode') != 'experimental'
            or not config['cache'].get('auto_write') or config['cache']['threshold'] != 0.9):
        raise ValueError('integrated candidate requires senior_v2, experimental FAQ/cache, 0.9 thresholds and FAQ margin 0.05')
    if 'speculative' in config:
        from .speculative import validate_speculative
        validate_speculative(config)
    return config


class IntegratedQA:
    def __init__(self, root, config):
        if 'speculative' in config:
            from .speculative import SpeculativeA0
            self.adapter = SpeculativeA0(root, config)
        else:
            self.adapter = A0Adapter(root, config)
        self.faq = FAQFallback(root, config, self.adapter)
        faq_identity = self.faq.identity['faq']
        hashes = dict(faq_identity['input_sha256'])
        if faq_identity['review']:
            hashes[faq_identity['review']['path']] = faq_identity['review']['sha256']
        # Pin routing code and FAQ resources as well as the existing A0 resources.
        hashes.update({str(p.resolve()): file_hash(p) for p in Path(__file__).parent.glob('*.py')})
        self.guard = ResourceGuard(self.adapter, extra_hashes=hashes)
        try:
            self.cache = ReviewedCacheA0(self.adapter, config['cache'], resource_guard=self.guard)
        except Exception:
            self.guard.close()
            raise
        self.identity = {'a0': self.adapter.identity, 'faq': faq_identity,
                         'cache': self.cache.identity, 'extra_sha256': hashes,
                         'formal_quality_certified': False}

    def answer(self, question):
        start = time.perf_counter()
        _, validation = measured(self.guard.verify)
        vector, embedding = measured(self.adapter.embed, question)
        (answer, faq), faq_seconds = measured(self.faq.match_only, question, query_vector=vector)
        timing = {'resource_validation_seconds': validation, 'embedding_seconds': embedding,
                  'faq_seconds': faq_seconds, 'cache_lookup_seconds': 0.,
                  'rag_llm_seconds': 0., 'cache_write_seconds': 0.}
        cache_detail = None
        if faq['faq_accepted']:
            source, verification = 'faq', 'unverified'
        else:
            answer, cache_detail = self.cache.answer(question, query_vector=vector)
            source, verification = cache_detail['answer_source'], cache_detail['answer_verification']
            for name, value in cache_detail['timings'].items():
                timing[name] = timing.get(name, 0.) + value
        speculative = {}
        if 'speculative' in self.adapter.config:
            from .speculative import unused_statistics
            speculative['speculative'] = ((cache_detail.get('rag_detail') or {}).get('speculative')
                                          if source == 'rag' else unused_statistics())
        return answer, {**speculative, 'answer_source': source, 'answer_verification': verification,
                        'layers': ['faq'] if source == 'faq' else ['faq', 'cache'] if source == 'cache'
                                  else ['faq', 'cache', 'rag', 'llm'],
                        'faq': faq, 'cache': cache_detail,
                        'cache_size': len(self.cache.cache.entries),
                        'timings': timing, 'total_seconds': time.perf_counter()-start}

    def close(self):
        self.guard.close()
        if hasattr(self.adapter, 'close'):
            self.adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', default='experiments/configs/integrated_baseline_v1.experimental.json')
    parser.add_argument('--run-dir', help='new child of experiments/results; defaults to a unique directory')
    parser.add_argument('--details', action='store_true', help='also print RAG document excerpts')
    args = parser.parse_args()
    config = load_config(args.config)
    raw = Path(args.run_dir) if args.run_dir else Path('experiments/results/integrated_cli') / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    output = (raw if raw.is_absolute() else ROOT / raw).resolve()
    if (ROOT / 'experiments/results').resolve() not in output.parents:
        parser.error('run-dir must be a new child of experiments/results')
    output.mkdir(parents=True, exist_ok=False)
    engine = None
    try:
        print('載入共用模型、Dense／FAQ索引與記憶體Cache……')
        engine = IntegratedQA(ROOT, config)
        with (output / 'session.json').open('x', encoding='utf-8') as stream:
            json.dump({'config': config, 'identity': engine.identity}, stream, ensure_ascii=False, indent=2)
        print('baseline_senior_v2；FAQ / Cache experimental，門檻尚未校準。')
        print('輸入 exit 結束；記錄位置:', output)
        with (output / 'results.jsonl').open('x', encoding='utf-8') as stream:
            while True:
                try:
                    question = input('您的問題: ').strip()
                    if question.lower() in ('exit', 'quit', 'q'):
                        break
                    if not question:
                        continue
                    started = time.perf_counter()
                    try:
                        answer, detail = engine.answer(question)
                        record = dict(status='success', question=question, answer=answer,
                                      answer_char_count=len(answer), exceeds_100_characters=len(answer)>100, **detail)
                        print('回答:', answer)
                        print('流程:', ' → '.join(detail['layers']), '來源:', detail['answer_source'])
                        faq, cache = detail['faq'], detail['cache']
                        print('FAQ分數:', faq['faq_top1_score'], '差距:', faq['faq_score_margin'],
                              '判斷:', faq['faq_acceptance_reason'] or faq['faq_rejection_reason'])
                        if cache is not None:
                            print('Cache:', 'Hit' if cache['cache_hit'] else 'Miss', cache['hit_type'],
                                  '分數:', cache['similarity'], '原因:', cache['reason'], '自動寫入:', cache['saved'])
                        else:
                            print('Cache: 未執行')
                        print('快取筆數:', detail['cache_size'], '答案狀態:', detail['answer_verification'],
                              '字數:', len(answer), '超過100字:', len(answer)>100)
                        print('總耗時: %.3f 秒；階段耗時:' % detail['total_seconds'], detail['timings'])
                        if args.details and cache:
                            print(json.dumps(cache.get('rag_detail'), ensure_ascii=False, indent=2))
                    except Exception as exc:
                        record = dict(status='failed', question=question, error=str(exc),
                                      total_seconds=time.perf_counter()-started)
                        print('失敗:', exc)
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                    stream.flush()
                except (EOFError, KeyboardInterrupt):
                    break
    finally:
        if engine is not None:
            engine.close()


if __name__ == '__main__':
    main()
