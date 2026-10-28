"""Robot-side HTTP client; Python standard library only, no model dependencies."""
import argparse
import json
import urllib.error
import urllib.request


def ask(question, base_url="http://127.0.0.1:8000", timeout=130):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat",
        data=json.dumps({"question": question}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
            if (not isinstance(result, dict) or type(result.get("success")) is not bool
                    or not isinstance(result.get("answer"), str)):
                return {"success": False, "answer": "", "error": "invalid_server_response"}
            return result
    except urllib.error.HTTPError as exc:
        try:
            result = json.loads(exc.read().decode("utf-8"))
            if not isinstance(result, dict):
                raise ValueError("non-object error response")
        except (ValueError, UnicodeError):
            result = {"success": False, "answer": "", "error": "http_error"}
        result["http_status"] = exc.code
        result.update(success=False, answer="")
        return result
    except (ValueError, UnicodeError):
        return {"success": False, "answer": "", "error": "invalid_server_response"}
    except (urllib.error.URLError, TimeoutError, OSError):
        # The robot must finish its waiting state even if the server is offline.
        return {"success": False, "answer": "", "error": "connection_failed_or_client_timeout"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--question", default="資料科學系有哪些學習資源？")
    args = parser.parse_args()
    print(json.dumps(ask(args.question, args.url), ensure_ascii=False, indent=2))
