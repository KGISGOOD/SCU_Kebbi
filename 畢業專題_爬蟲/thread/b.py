from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import unquote
import pandas as pd
import json
import re
import os
import html as ihtml


# ============================================================
# 基本設定
# ============================================================

THREADS_URL = "https://www.threads.com/search?q=%E6%9D%B1%E5%90%B3%E8%B3%87%E6%96%99%E7%A7%91%E5%AD%B8%E7%B3%BB&serp_type=default&hl=zh-tw"

STATE_FILE = "threads_state.json"

SCROLL_TIMES = 100
WAIT_SECONDS = 5

OUTPUT_TXT = "東吳資料科學系_threads_texts.txt"
OUTPUT_CSV = "東吳資料科學系_threads_texts.csv"
OUTPUT_HTML = "東吳資料科學系_threads_raw.html"
OUTPUT_JSON_TEXTS = "東吳資料科學系_threads_extracted_texts_debug.json"

# 你的搜尋關鍵字，可自行增加
KEYWORDS = [
    "東吳",
    "資料科學",
    "資料科學系",
    "資科",
    "東吳資科",
    "會計學系",
    "財務工程",
    "精算數學"
]


# ============================================================
# 文字清理
# ============================================================

def clean_text_keep_newlines(text):
    """
    保留貼文換行，但清掉多餘空白。
    """
    if text is None:
        return ""

    text = str(text)
    text = ihtml.unescape(text)
    text = text.replace("\\n", "\n")
    text = text.replace("\\/", "/")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")

    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
        if line:
            lines.append(line)

    text = "\n".join(lines).strip()
    return text


def clean_text_one_line(text):
    """
    用於比較去重。
    """
    text = clean_text_keep_newlines(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_cjk(text):
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def contains_keyword(text):
    return any(k in text for k in KEYWORDS)


def is_url_like(text):
    return bool(re.search(r"https?://|www\.|\.com|\.net|\.org", text))


def is_probably_ui_text(text):
    """
    過濾 Threads UI、按鈕、導覽列、政策文字。
    """
    one = clean_text_one_line(text)

    if not one:
        return True

    blacklist_exact = {
        "Threads",
        "搜尋",
        "登入",
        "註冊",
        "首頁",
        "通知",
        "個人檔案",
        "更多",
        "回覆",
        "轉發",
        "引用",
        "分享",
        "讚",
        "查看翻譯",
        "顯示更多",
        "發佈",
        "Instagram",
        "Meta",
        "使用條款",
        "隱私政策",
        "Cookie",
        "允許所有",
        "拒絕非必要",
        "完成",
        "取消",
        "下一步",
        "返回",
        "正在載入",
        "Loading",
        "Log in",
        "Sign up",
        "Search",
        "For you",
        "Following",
        "Replies",
        "Reposts",
        "Quotes",
        "Likes",
        "Post",
        "Repost",
        "Quote",
        "Like",
        "Reply",
        "Share",
    }

    blacklist_contains = [
        "登入即可查看",
        "註冊即可查看",
        "使用 Threads",
        "透過 Threads",
        "加入 Threads",
        "使用 Instagram 帳號",
        "改以用戶名稱登入",
        "來自 Instagram",
        "Meta Platforms",
        "隱私政策",
        "Cookie 政策",
        "Threads 使用條款",
        "Terms",
        "Privacy",
        "Help Center",
        "查無結果",
        "查看人們談論的主題",
        "暢所欲言",
    ]

    if one in blacklist_exact:
        return True

    for word in blacklist_contains:
        if word in one:
            return True

    # 純數字、日期、時間、符號
    if re.fullmatch(r"[\d\s年月日分鐘小時天週周·:./,\-]+", one):
        return True

    # 很短純英文通常是 UI
    if re.fullmatch(r"[A-Za-z\s]+", one) and len(one) < 40:
        return True

    # URL 或資源路徑通常不是貼文
    if is_url_like(one) and len(one) < 120:
        return True

    # CSS/JS 痕跡
    if any(x in one for x in ["function(", "var ", "window.", "document.", "rgb(", "rgba(", "--barcelona"]):
        return True

    return False


def looks_like_post_text(text):
    """
    判斷文字是否像 Threads 貼文內文。
    """
    text = clean_text_keep_newlines(text)
    one = clean_text_one_line(text)

    if not one:
        return False

    if is_probably_ui_text(one):
        return False

    # 太短通常不是貼文
    if len(one) < 8:
        return False

    # 太長通常是整頁 JSON 或混雜資料
    if len(one) > 1200:
        return False

    # 至少要有中文
    if not contains_cjk(one):
        return False

    # 搜尋頁目標：優先保留含關鍵字的內容
    # 但有些內文可能只出現「有人可以給建議嗎」這類文字，
    # 所以如果句子夠像貼文，也保留。
    if contains_keyword(one):
        return True

    post_like_patterns = [
        "有人",
        "請問",
        "想問",
        "可以給",
        "建議",
        "推薦",
        "原因",
        "系",
        "大學",
        "科系",
        "認親",
    ]

    if any(p in one for p in post_like_patterns) and len(one) >= 12:
        return True

    return False


# ============================================================
# 從 HTML meta 抽內文
# ============================================================

def extract_from_meta(html):
    """
    抽取 og:description、description、twitter:description。
    這對 Threads 單篇貼文或搜尋保存頁很重要。
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    meta_keys = {
        "description",
        "og:description",
        "twitter:description",
        "title",
        "og:title",
        "twitter:title",
    }

    for meta in soup.find_all("meta"):
        key = meta.get("property") or meta.get("name")
        content = meta.get("content")

        if not key or not content:
            continue

        if key in meta_keys:
            text = clean_text_keep_newlines(content)
            if looks_like_post_text(text):
                results.append({
                    "source": f"meta:{key}",
                    "text": text
                })

    return results


# ============================================================
# 從 DOM 抽內文
# ============================================================

def extract_from_html_dom(html):
    """
    從已渲染 HTML 抽可能貼文文字。
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "img", "video", "button"]):
        tag.decompose()

    results = []

    selectors = [
        "article",
        "[role='article']",
        "div[dir='auto']",
        "span[dir='auto']",
        "p",
        "a[href*='/post/']",
        "div",
        "span",
    ]

    seen_elements = set()

    for selector in selectors:
        for tag in soup.select(selector):
            ident = id(tag)
            if ident in seen_elements:
                continue
            seen_elements.add(ident)

            text = tag.get_text("\n", strip=True)
            text = clean_text_keep_newlines(text)

            if looks_like_post_text(text):
                results.append({
                    "source": f"dom:{selector}",
                    "text": text
                })

    return results


def extract_visible_texts_from_page(page):
    """
    直接從瀏覽器渲染後的可見元素抽文字。
    這比單純 page.content() 有時更準。
    """
    js = """
    () => {
        const selectors = [
            "article",
            "[role='article']",
            "div[dir='auto']",
            "span[dir='auto']",
            "p",
            "a[href*='/post/']"
        ];

        const texts = [];

        for (const selector of selectors) {
            const nodes = Array.from(document.querySelectorAll(selector));

            for (const el of nodes) {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);

                if (
                    rect.width <= 0 ||
                    rect.height <= 0 ||
                    style.visibility === "hidden" ||
                    style.display === "none"
                ) {
                    continue;
                }

                const text = el.innerText || el.textContent || "";
                if (text.trim()) {
                    texts.push({
                        source: "visible:" + selector,
                        text: text
                    });
                }
            }
        }

        return texts;
    }
    """

    try:
        raw = page.evaluate(js)
    except Exception:
        raw = []

    results = []

    for item in raw:
        text = clean_text_keep_newlines(item.get("text", ""))
        if looks_like_post_text(text):
            results.append({
                "source": item.get("source", "visible"),
                "text": text
            })

    return results


# ============================================================
# 從 JSON / Network response 抽內文
# ============================================================

TEXT_KEYS = {
    "text",
    "caption",
    "description",
    "body",
    "title",
    "message",
    "content",
    "post_text",
    "thread_text",
}


def walk_json(obj, path=""):
    """
    遞迴走訪 JSON，抽出可能是貼文文字的字串。
    """
    results = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else str(k)

            if isinstance(v, str):
                text = clean_text_keep_newlines(v)

                # key 是文字相關欄位時，優先檢查
                if str(k).lower() in TEXT_KEYS and looks_like_post_text(text):
                    results.append({
                        "source": f"json:{new_path}",
                        "text": text
                    })
                else:
                    # 有些 Threads JSON key 不固定，所以也掃所有 string
                    if looks_like_post_text(text):
                        results.append({
                            "source": f"json:{new_path}",
                            "text": text
                        })

            elif isinstance(v, (dict, list)):
                results.extend(walk_json(v, new_path))

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            results.extend(walk_json(item, f"{path}[{i}]"))

    elif isinstance(obj, str):
        text = clean_text_keep_newlines(obj)
        if looks_like_post_text(text):
            results.append({
                "source": f"json:{path}",
                "text": text
            })

    return results


def extract_json_from_script_tags(html):
    """
    Threads / Meta 網頁常把 ServerJS payload 塞在 script[type='application/json']。
    這裡會盡量 JSON parse 後遞迴抽文字。
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for script in soup.find_all("script"):
        script_type = script.get("type", "")
        content = script.string or script.get_text()

        if not content:
            continue

        content = content.strip()

        # 只處理看起來像 JSON 的 script
        if "application/json" not in script_type and not content.startswith("{") and not content.startswith("["):
            continue

        try:
            data = json.loads(content)
            results.extend(walk_json(data, "script_json"))
        except Exception:
            # 有些 script 不是純 JSON，略過
            continue

    return results


# ============================================================
# 去重與後處理
# ============================================================

def dedupe_results(results):
    """
    去重，保留換行版本。
    """
    cleaned = []

    for item in results:
        text = clean_text_keep_newlines(item.get("text", ""))
        if not looks_like_post_text(text):
            continue

        one = clean_text_one_line(text)

        cleaned.append({
            "source": item.get("source", ""),
            "text": text,
            "key": one
        })

    # 先去完全重複
    seen = set()
    unique = []

    for item in cleaned:
        if item["key"] in seen:
            continue
        seen.add(item["key"])
        unique.append(item)

    # 移除「被更長文字包含」的短片段
    final = []

    for i, item in enumerate(unique):
        text_key = item["key"]
        is_substring = False

        for j, other in enumerate(unique):
            if i == j:
                continue

            other_key = other["key"]

            if len(text_key) < len(other_key) and text_key in other_key:
                is_substring = True
                break

        if not is_substring:
            final.append({
                "source": item["source"],
                "text": item["text"]
            })

    return final


# ============================================================
# 登入狀態處理
# ============================================================

def ensure_login_state():
    """
    如果沒有 threads_state.json，先開瀏覽器讓你手動登入一次。
    """
    if os.path.exists(STATE_FILE):
        print(f"已找到登入狀態：{STATE_FILE}")
        return

    print(f"找不到 {STATE_FILE}，將開啟瀏覽器讓你手動登入 Threads。")

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                headless=False,
                channel="chrome",
                args=["--disable-blink-features=AutomationControlled"]
            )
        except Exception:
            browser = p.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled"]
            )

        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="zh-TW",
            timezone_id="Asia/Taipei",
        )

        page = context.new_page()
        page.goto("https://www.threads.com/", wait_until="domcontentloaded", timeout=90000)

        print("\n請在打開的瀏覽器中手動登入 Threads / Instagram。")
        print("登入完成後，確認可以看到 Threads 首頁或搜尋頁。")
        input("登入完成後，回到這裡按 Enter 繼續...")

        context.storage_state(path=STATE_FILE)
        print(f"登入狀態已儲存到：{STATE_FILE}")

        browser.close()


# ============================================================
# 主爬蟲
# ============================================================

def crawl_threads_search():
    captured_json_results = []

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(
                headless=False,
                channel="chrome",
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                ]
            )
        except Exception:
            browser = p.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                ]
            )

        context = browser.new_context(
            storage_state=STATE_FILE if os.path.exists(STATE_FILE) else None,
            viewport={"width": 1280, "height": 900},
            locale="zh-TW",
            timezone_id="Asia/Taipei",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        page = context.new_page()

        # 攔截 network JSON response
        def on_response(response):
            url = response.url.lower()
            content_type = response.headers.get("content-type", "").lower()

            interesting = any(x in url for x in [
                "graphql",
                "api",
                "query",
                "search",
                "ajax",
                "barcelona"
            ])

            if not interesting:
                return

            try:
                if "json" in content_type or "application/x-javascript" in content_type or "text/javascript" in content_type:
                    try:
                        data = response.json()
                        extracted = walk_json(data, "network")
                        captured_json_results.extend(extracted)
                    except Exception:
                        try:
                            txt = response.text()
                            txt_clean = clean_text_keep_newlines(txt)
                            if looks_like_post_text(txt_clean):
                                captured_json_results.append({
                                    "source": f"network_text:{response.url}",
                                    "text": txt_clean
                                })
                        except Exception:
                            pass
            except Exception:
                pass

        page.on("response", on_response)

        print("正在開啟 Threads 搜尋頁...")
        page.goto(THREADS_URL, wait_until="domcontentloaded", timeout=90000)

        page.wait_for_timeout(8000)

        # 如果遇到登入牆，讓你有機會手動處理
        body_text = ""
        try:
            body_text = page.locator("body").inner_text(timeout=5000)
        except Exception:
            pass

        if "登入" in body_text and "使用 Instagram 帳號" in body_text:
            print("\n偵測到可能還在登入頁或登入牆。")
            print("請在瀏覽器中確認是否已登入。")
            input("如果你已經手動登入完成，請按 Enter 繼續爬取...")

            context.storage_state(path=STATE_FILE)
            print(f"已更新登入狀態：{STATE_FILE}")

        print(f"\n開始滾動，共 {SCROLL_TIMES} 次")

        visible_results = []

        for i in range(SCROLL_TIMES):
            # 先抽目前畫面可見文字
            visible_results.extend(extract_visible_texts_from_page(page))

            # 滾動
            page.mouse.wheel(0, 5000)
            page.wait_for_timeout(WAIT_SECONDS * 1000)

            current_height = page.evaluate("document.body.scrollHeight")
            print(f"已完成第 {i + 1} 次滾動，頁面高度：{current_height}")

        # 最後再抽一次可見文字
        visible_results.extend(extract_visible_texts_from_page(page))

        html = page.content()

        with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
            f.write(html)

        browser.close()

    print("\nHTML 已儲存。")

    all_results = []

    # 1. 從 meta 抽
    all_results.extend(extract_from_meta(html))

    # 2. 從 DOM HTML 抽
    all_results.extend(extract_from_html_dom(html))

    # 3. 從瀏覽器 visible text 抽
    all_results.extend(visible_results)

    # 4. 從 script JSON 抽
    all_results.extend(extract_json_from_script_tags(html))

    # 5. 從 network JSON response 抽
    all_results.extend(captured_json_results)

    final_results = dedupe_results(all_results)

    return final_results


# ============================================================
# 執行
# ============================================================

if __name__ == "__main__":
    ensure_login_state()

    results = crawl_threads_search()

    print(f"\n共抽取到 {len(results)} 筆可能的 Threads 貼文內文\n")

    for i, item in enumerate(results, 1):
        print(f"{i}. [{item['source']}]")
        print(item["text"])
        print("-" * 80)

    # 輸出 TXT
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for i, item in enumerate(results, 1):
            f.write(f"{i}. {item['text']}\n\n")

    # 輸出 CSV
    df = pd.DataFrame({
        "id": range(1, len(results) + 1),
        "source": [item["source"] for item in results],
        "text": [item["text"] for item in results],
    })

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 輸出 debug JSON
    with open(OUTPUT_JSON_TEXTS, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"已輸出：{OUTPUT_TXT}")
    print(f"已輸出：{OUTPUT_CSV}")
    print(f"已輸出：{OUTPUT_HTML}")
    print(f"已輸出：{OUTPUT_JSON_TEXTS}")

    print("\n前 20 筆：")
    print(df.head(20))