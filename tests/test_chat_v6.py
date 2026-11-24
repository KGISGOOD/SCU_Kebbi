"""HTTP and console mocks only; never contact an API or load a model."""
import importlib.util
import http.client
import io
import json
from pathlib import Path
import socket
import unittest
import urllib.error
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location('chat_v6', Path(__file__).with_name('chat_v6.py'))
chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat)


def response(body, status=200):
    result = Mock()
    result.getcode.return_value = status
    result.read.return_value = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode('utf-8')
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


class ChatV6Tests(unittest.TestCase):
    def test_utf8_question_only_and_verbatim_answer(self):
        answer = '完整繁體中文，包含「引號」、逗號,\n及第二行。'
        question = '  東吳資科的課程有哪些？  '
        with patch.object(chat.urllib.request, 'urlopen', return_value=response(
                dict(success=True, answer=answer, answer_source='rag', latency_seconds=1.23456))) as opened:
            result = chat.ask(question)
        request = opened.call_args[0][0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:8000/chat')
        self.assertEqual(request.get_method(), 'POST')
        self.assertEqual(json.loads(request.data.decode('utf-8')), {'question': question})
        self.assertIn('charset=utf-8', request.get_header('Content-type'))
        self.assertEqual(result['answer'], answer)
        self.assertEqual(opened.call_count, 1)

    def test_offline_and_direct_or_wrapped_timeout(self):
        cases = [(urllib.error.URLError(ConnectionRefusedError()), '無法連線'),
                 (TimeoutError(), '請求逾時'),
                 (socket.timeout(), '請求逾時'),
                 (urllib.error.URLError(TimeoutError()), '請求逾時'),
                 (urllib.error.URLError(socket.timeout()), '請求逾時'),
                 (http.client.IncompleteRead(b'partial'), '連線中斷'),
                 (ConnectionResetError('reset'), '連線中斷')]
        for error, text in cases:
            with self.subTest(error=type(error).__name__), \
                    patch.object(chat.urllib.request, 'urlopen', side_effect=error) as opened:
                with self.assertRaisesRegex(chat.ChatError, text):
                    chat.ask('問題')
                self.assertEqual(opened.call_count, 1)

    def test_http_errors_never_return_body_as_answer(self):
        for status in (422, 500, 503, 504):
            error = urllib.error.HTTPError('http://fixture/chat', status, 'error', {},
                                          io.BytesIO(b'{"success":false,"answer":"never serve","error":"busy"}'))
            with self.subTest(status=status), patch.object(chat.urllib.request, 'urlopen', side_effect=error):
                with self.assertRaisesRegex(chat.ChatError, 'HTTP %d' % status) as caught:
                    chat.ask('問題')
                self.assertNotIn('never serve', str(caught.exception))
                if status == 504:
                    self.assertIn('Cache 狀態可能已改變', str(caught.exception))

    def test_non_json_error_body_and_non_200_response(self):
        error = urllib.error.HTTPError('http://fixture/chat', 500, 'error', {}, io.BytesIO(b'<html>error</html>'))
        with patch.object(chat.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(chat.ChatError, 'HTTP 500'):
                chat.ask('問題')
        with patch.object(chat.urllib.request, 'urlopen', return_value=response({}, status=201)):
            with self.assertRaisesRegex(chat.ChatError, '非200'):
                chat.ask('問題')

    def test_invalid_json_utf8_success_and_answer(self):
        for body in (b'not-json', b'\xff', [], {'success': False, 'answer': '不是答案'},
                     {'success': 1, 'answer': '不是答案'}, {'success': True, 'answer': ''},
                     {'success': True, 'answer': None}, {'success': True, 'answer': '  '},
                     {'success': True, 'answer': '不是答案', 'error': 'failure'}):
            with self.subTest(body=body), patch.object(chat.urllib.request, 'urlopen', return_value=response(body)):
                with self.assertRaises(chat.ChatError):
                    chat.ask('問題')

    def test_interaction_continues_after_error_and_preserves_output(self):
        output, errors = io.StringIO(), io.StringIO()
        answer = '  回答的空白也保留。\n完整第二行。  '
        with patch('builtins.input', side_effect=['', '問題一', '問題二', '退出']), \
                patch.object(chat, 'ask', side_effect=[chat.ChatError('離線'),
                    dict(answer=answer, answer_source='cache', latency_seconds=.01234)]) as ask, \
                patch('sys.stdout', output), patch('sys.stderr', errors):
            chat.interact('http://fixture', 20)
        self.assertEqual([call.args[0] for call in ask.call_args_list], ['問題一', '問題二'])
        self.assertIn('回答：\n' + answer + '\n', output.getvalue())
        self.assertIn('來源：cache｜API 耗時：0.0123 秒', output.getvalue())
        self.assertIn('離線', errors.getvalue())

    def test_exit_commands_eof_and_interrupt_send_no_requests(self):
        for command in ('exit', 'quit', '退出', ' EXIT ', EOFError(), KeyboardInterrupt()):
            with self.subTest(command=command), patch('builtins.input', side_effect=[command]), \
                    patch.object(chat, 'ask') as ask, patch('sys.stdout', io.StringIO()):
                chat.interact('http://fixture', 20)
                ask.assert_not_called()

    def test_utf8_stream_configuration(self):
        streams = [Mock(), Mock(), Mock()]
        with patch.object(chat.sys, 'stdin', streams[0]), patch.object(chat.sys, 'stdout', streams[1]), \
                patch.object(chat.sys, 'stderr', streams[2]):
            chat.configure_utf8()
        for stream in streams:
            stream.reconfigure.assert_called_once_with(encoding='utf-8', errors='strict')


if __name__ == '__main__':
    unittest.main()
