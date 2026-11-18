"""Sequential V6 HTTP runner. Reads authoritative API history; never writes it."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import urllib.request
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
FILES = [('A', 'A_standard.csv', 25), ('B', 'B_similar_same.csv', 25),
         ('C', 'C_similar_different.csv', 25), ('D', 'D_no_answer.csv', 15),
         ('E', 'E_unrelated.csv', 10), ('F', 'F_ambiguous.csv', 20)]
HISTORY_HEADER = ['問題', '答案', '回答來源', 'Embedding 耗時（秒）', 'FAQ 耗時（秒）',
                  'Cache 耗時（秒）', 'RAG 檢索耗時（秒）', 'SD 生成耗時（秒）',
                  'Greedy 驗證耗時（秒）', '資源驗證耗時（秒）', '總耗時（秒）', 'Output Tokens']
HEADER = ['問題分類'] + HISTORY_HEADER


def questions(directory, category=None):
    plan = []
    for label, filename, count in FILES:
        rows = read_csv(Path(directory) / filename, ['問題'])
        if len(rows) != count or any(not r[0].strip() for r in rows):
            raise ValueError('題數或問題內容不符: ' + filename)
        if category is None or label == category:
            plan.extend((label, r[0]) for r in rows)
    return plan


def read_csv(path, header):
    with Path(path).open('rb') as f:
        raw = f.read()
    if not raw.startswith(b'\xef\xbb\xbf') or not raw.endswith(b'\n'):
        raise ValueError('CSV 缺少 BOM 或末列不完整: ' + str(path))
    rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True))
    if not rows or rows[0] != header or any(len(r) != len(header) for r in rows[1:]):
        raise ValueError('CSV 欄位不相容: ' + str(path))
    return rows[1:]


def validate_record(row):
    if len(row) != 12 or not row[0].strip() or not row[1].strip() or row[2] not in ('FAQ', 'Cache', 'RAG + SD'):
        raise ValueError('不完整或不成功的歷史紀錄')
    for value in row[3:]:
        if value and (not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError('無效計時或 Token 數')
    if row[2] in ('FAQ', 'Cache') and any(row[i] for i in (6, 7, 8, 11)):
        raise ValueError('FAQ/Cache 不應有新的生成紀錄')
    if row[11] and (not row[11].isdigit()):
        raise ValueError('Output Tokens 必須為非負整數或空白')


def http_json(base_url, method, endpoint, timeout, payload=None):
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(base_url.rstrip('/') + endpoint, data=data,
                                     headers={'Content-Type': 'application/json; charset=utf-8'}, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError('API 未回傳 JSON 物件')
    return value


def write_state(path, state):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(state, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


@contextmanager
def exclusive(path):
    """A separate OS lock prevents two runners from appending the same result."""
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(plan, output, history, base_url, timeout=180, resume=False, restart=False,
        retry_uncertain=False, client=http_json):
    output, history = Path(output).resolve(), Path(history).resolve()
    if output == history or output.with_suffix('.state.json') == history:
        raise ValueError('結果路徑不可指向 API 歷史紀錄')
    if resume and restart:
        raise ValueError('resume 與 restart 不可同時使用')
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.with_suffix('.state.json')
    with exclusive(output.with_suffix('.lock')):
        digest = hashlib.sha256(json.dumps(plan, ensure_ascii=False).encode('utf-8')).hexdigest()
        expected = {'plan_sha256': digest, 'base_url': base_url, 'history_file': str(history)}
        if output.exists() and not (resume or restart):
            raise ValueError('結果已存在：使用 --resume 續跑，或 --restart 明確重測並覆寫結果')
        if restart:
            print('警告：重測會覆寫本執行器結果，但不會清空 API Cache 或歷史；不是 Cold Cache 測試。', flush=True)
        if resume:
            print('續跑只驗證題目與結果前綴；無法確認 API 是否曾重啟或 Cache 是否曾被其他用戶改變。', flush=True)
            saved = read_csv(output, HEADER)
            state = json.loads(state_path.read_text(encoding='utf-8'))
            if any(state.get(k) != v for k, v in expected.items()):
                raise ValueError('題庫、分類、API 或歷史路徑變動，拒絕混用結果')
            if len(saved) > len(plan):
                raise ValueError('結果筆數超過題庫')
            for i, row in enumerate(saved):
                if row[:2] != list(plan[i]):
                    raise ValueError('續跑問題分類或順序不符，第 %d 筆' % (i+1))
                validate_record(row[1:])
            pending = state.get('pending')
            if pending is not None:
                if pending == len(saved)-1:
                    state['pending'] = None  # CSV was durable before checkpoint completion.
                elif pending != len(saved):
                    raise ValueError('請求檢查點與結果不一致')
                elif not retry_uncertain:
                    raise ValueError('上次請求狀態不明，Cache 可能已改變；確認後用 --resume --retry-uncertain 重試該題')
                else:
                    print('警告：重試狀態不明請求，可能重複生成／改變 Cache，不能視為未中斷的效能測量。', flush=True)
        else:
            if state_path.exists() and not restart:
                raise ValueError('已有檢查點但缺少結果，請檢查或使用 --restart')
            saved, state = [], dict(expected, pending=None)
            with output.open('w' if restart else 'x', encoding='utf-8-sig', newline='') as stream:
                csv.writer(stream).writerow(HEADER)
                stream.flush()
                os.fsync(stream.fileno())
        write_state(state_path, state)
        if len(saved) == len(plan):
            print('所有題目已完成，未呼叫 API。', flush=True)
            return len(saved)
        health = client(base_url, 'GET', '/health', min(timeout, 15))
        if (health.get('status') != 'ready' or health.get('api_mode') != 'integrated_speculative'
                or health.get('models_ready') is not True or health.get('accepting_requests') is not True):
            raise ValueError('V6 API 尚未 ready 或仍忙碌，停止測試')
        with output.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            for index in range(len(saved), len(plan)):
                category, question = plan[index]
                before = read_csv(history, HISTORY_HEADER)
                state['pending'] = index
                write_state(state_path, state)  # Must precede sending the non-idempotent POST.
                try:
                    response = client(base_url, 'POST', '/chat', timeout, {'question': question})
                    if response.get('success') is not True or response.get('error'):
                        raise ValueError('API 問答失敗: ' + str(response.get('error')))
                    answer = response.get('answer')
                    if not isinstance(answer, str) or not answer.strip() or not response.get('request_id'):
                        raise ValueError('API 成功回應不完整')
                    after = read_csv(history, HISTORY_HEADER)
                    if len(after) != len(before)+1 or after[:-1] != before:
                        raise ValueError('歷史 CSV 未恰好追加一筆或有其他請求，拒絕配對')
                    row = after[-1]
                    source = {'faq': 'FAQ', 'cache': 'Cache', 'rag': 'RAG + SD'}.get(response.get('answer_source'))
                    validate_record(row)
                    if row[:3] != [question, answer, source]:
                        raise ValueError('API 回應與歷史 CSV 不一致')
                    offset = stream.tell()
                    try:
                        writer.writerow([category] + row)
                        stream.flush()
                        os.fsync(stream.fileno())
                    except Exception:
                        stream.seek(offset)
                        stream.truncate()
                        raise
                    state['pending'] = None
                    write_state(state_path, state)
                    print('[%d/%d] %s 完成｜%s｜%s 秒｜%s' %
                          (index+1, len(plan), category, source, row[10], question), flush=True)
                except BaseException:
                    print('已停止，成功資料已保留。請求可能已執行；尤其 timeout 後 Cache 狀態可能已改變。不要直接重送。', file=sys.stderr)
                    raise
        return len(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--category', choices=list('ABCDEF'))
    parser.add_argument('--history-file', type=Path, default=ROOT/'experiments/results/api_logs/v6_chat_history.csv',
                        help='API 實際歷史 CSV；遠端 API 必須提供可讀取的共用路徑')
    parser.add_argument('--timeout', type=float, default=180)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true')
    group.add_argument('--restart', action='store_true', help='覆寫本執行器結果，重新提問；不清空 API Cache 或歷史')
    parser.add_argument('--retry-uncertain', action='store_true', help='明確允許重試上一個狀態不明請求，須搭配 resume')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0 or (args.retry_uncertain and not args.resume):
        parser.error('timeout 必須為正數；retry-uncertain 必須搭配 resume')
    try:
        run(questions(ROOT/'tests/questions', args.category), ROOT/'tests/results/v6_test_results.csv',
            args.history_file, args.base_url, args.timeout, args.resume, args.restart, args.retry_uncertain)
    except (Exception, KeyboardInterrupt) as exc:
        parser.exit(1, '測試停止: %s\n' % exc)


if __name__ == '__main__':
    main()
