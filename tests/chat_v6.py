"""Interactive, single-turn HTTP client for the existing V6 /chat API."""
import argparse
import http.client
import json
import math
import socket
import sys
import urllib.error
import urllib.request


class ChatError(Exception):
    """A request failed; its error must never be displayed as a model answer."""


def configure_utf8():
    # Windows console uses Python's Unicode console support. Redirected streams
    # use explicit UTF-8 too; do not alter the process-wide console code page.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='strict')


def timeout_error():
    return ChatError('請求逾時；API 可能仍在運算，Cache 狀態可能已改變。請勿立即重送。')


def ask(question, base_url='http://127.0.0.1:8000', timeout=180):
    request = urllib.request.Request(
        base_url.rstrip('/') + '/chat',
        data=json.dumps({'question': question}, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json; charset=utf-8'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.getcode() != 200:
                raise ChatError('API 回傳非200狀態：HTTP %s。' % response.getcode())
            result = json.loads(response.read().decode('utf-8-sig'))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode('utf-8-sig'))
            detail = body.get('error') if isinstance(body, dict) else None
        except (ValueError, UnicodeError, OSError):
            detail = None
        finally:
            exc.close()
        message = 'API 回傳 HTTP %s%s。' % (exc.code, '：' + detail if isinstance(detail, str) else '')
        if exc.code == 504:
            message += 'API 可能仍在運算，Cache 狀態可能已改變。請勿立即重送。'
        raise ChatError(message) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise timeout_error() from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            raise timeout_error() from exc
        raise ChatError('無法連線到 V6 API；請確認服務已啟動，且網址與連接埠正確。') from exc
    except (OSError, http.client.HTTPException) as exc:
        raise ChatError('API 連線中斷：%s。請確認服務狀態；本工具不會自動重試。' % exc) from exc
    except (ValueError, UnicodeError) as exc:
        raise ChatError('API 回傳的資料不是有效的 UTF-8 JSON。') from exc
    if not isinstance(result, dict) or result.get('success') is not True or result.get('error'):
        detail = result.get('error') if isinstance(result, dict) else None
        raise ChatError('API 未成功完成問答%s。' % ('：' + detail if isinstance(detail, str) else ''))
    if not isinstance(result.get('answer'), str) or not result['answer'].strip():
        raise ChatError('API 回傳的答案為空或格式不正確。')
    return result


def interact(base_url, timeout):
    print('V6 單輪問答｜輸入 exit、quit 或 退出 結束。')
    while True:
        try:
            question = input('問題：')
        except (EOFError, KeyboardInterrupt):
            print('\n已結束。')
            return
        if question.strip().casefold() in ('exit', 'quit', '退出'):
            return
        if not question.strip():
            continue
        try:
            result = ask(question, base_url, timeout)
        except ChatError as exc:
            print('錯誤：' + str(exc), file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print('\n已停止等待；API 可能仍在運算，Cache 狀態可能已改變。', file=sys.stderr)
            return
        print('回答：')
        print(result['answer'])  # Preserve the complete server answer verbatim.
        metadata = []
        if isinstance(result.get('answer_source'), str):
            metadata.append('來源：' + result['answer_source'])
        seconds = result.get('latency_seconds')
        if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
            metadata.append('API 耗時：%.4f 秒' % seconds)
        if metadata:
            print('｜'.join(metadata))


def main():
    configure_utf8()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout 必須為正數且有限')
    interact(args.url, args.timeout)


if __name__ == '__main__':
    main()
