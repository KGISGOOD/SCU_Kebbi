"""Output-only timing formatting; inference measurements retain full precision."""
import json
import math


def seconds_text(value):
    if value is None or value == '':
        return ''
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError('timing must be finite, non-negative seconds')
    return format(number, '.4f')


def timing_json(value, key=''):
    """Keep JSON timing numbers numeric, including four decimal places on the wire."""
    if isinstance(value, dict):
        return '{' + ','.join(json.dumps(k, ensure_ascii=False) + ':' + timing_json(v, k)
                              for k, v in value.items()) + '}'
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(timing_json(v, key) for v in value) + ']'
    if key.endswith('_seconds') and value is not None:
        return seconds_text(value)
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def executed_stage_timings(detail):
    """Mask unused stages at output only; zero-duration executed stages remain valid."""
    stages = dict(detail.get('timings') or {})
    cache = detail.get('cache')
    if cache is None:
        for key in ('cache_lookup_seconds', 'cache_write_seconds', 'rag_llm_seconds'):
            stages[key] = None
    else:
        if detail['answer_source'] != 'rag':
            stages['rag_llm_seconds'] = None
        if cache.get('save_status') in (None, 'not_applicable', 'ineligible', 'awaiting_explicit_human_review'):
            stages['cache_write_seconds'] = None
    return stages
