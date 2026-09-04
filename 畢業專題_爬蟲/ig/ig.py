import os
import re
import json
import time
import html as ihtml
from datetime import datetime
import pandas as pd
from playwright.sync_api import sync_playwright

# ============================================================
# 基本設定
# ============================================================
TARGET_URL = "https://www.instagram.com/scu_data_science/"
STATE_FILE = "instagram_state.json"

# 滾動次數與每次滾動間隔秒數
SCROLL_TIMES = 25
WAIT_SECONDS = 3.5

# 輸出檔案名稱
OUTPUT_CSV = "東吳資科_IG貼文.csv"
OUTPUT_TXT = "東吳資科_IG貼文.txt"


# ============================================================
# 文字清理工具
# ============================================================
def clean_post_text(text):
    """
    保留貼文換行與表情符號，去除多餘空白。
    """
    if not text:
        return ""
    text = str(text)
    text = ihtml.unescape(text)
    text = text.replace("\\n", "\n").replace("\\/", "/")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n")

    lines = [line.strip() for line in text.splitlines()]
    # 去除連續超過兩行的空白行
    cleaned_lines = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# ============================================================
# 從 Instagram API / GraphQL JSON 遞迴提取貼文內文
# ============================================================
def extract_posts_from_json(obj):
    """
    遞迴走訪 Instagram 前端 API 回傳的 JSON，
    精準抽取 shortcode、內文 (caption)、時間 (timestamp)。
    """
    posts = []

    def recurse(node):
        if isinstance(node, dict):
            shortcode = node.get("shortcode") or node.get("code")
            caption_text = None

            # 格式 1: edge_media_to_caption.edges[0].node.text
            if "edge_media_to_caption" in node and isinstance(node["edge_media_to_caption"], dict):
                edges = node["edge_media_to_caption"].get("edges", [])
                if edges and isinstance(edges, list):
                    first_node = edges[0].get("node", {})
                    caption_text = first_node.get("text")

            # 格式 2: caption.text
            elif "caption" in node:
                cap = node.get("caption")
                if isinstance(cap, dict):
                    caption_text = cap.get("text")
                elif isinstance(cap, str):
                    caption_text = cap

            if caption_text and isinstance(caption_text, str) and len(caption_text.strip()) > 0:
                raw_time = node.get("taken_at_timestamp") or node.get("taken_at")
                date_str = ""
                if raw_time:
                    try:
                        date_str = datetime.fromtimestamp(int(raw_time)).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        date_str = ""

                cleaned = clean_post_text(caption_text)
                if cleaned:
                    posts.append({
                        "shortcode": shortcode or "",
                        "url": f"https://www.instagram.com/p/{shortcode}/" if shortcode else "",
                        "date": date_str,
                        "text": cleaned
                    })

            for val in node.values():
                recurse(val)

        elif isinstance(node, list):
            for item in node:
                recurse(item)

    recurse(obj)
    return posts


# ============================================================
# 確保登入狀態
# ============================================================
def ensure_login_state():
    """
    若本地無 instagram_state.json，自動打開瀏覽器讓使用者手動登入一次並保存 Cookie。
    """
    if os.path.exists(STATE_FILE):
        print(f"[*] 找到既有登入狀態檔：{STATE_FILE}")
        return

    print(f"[*] 找不到登入狀態檔 {STATE_FILE}，將開啟瀏覽器進行手動登入...")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            locale="zh-TW",
            timezone_id="Asia/Taipei"
        )
        page = context.new_page()
        page.goto("https://www.instagram.com/accounts/login/", wait_until="domcontentloaded")

        print("\n========================================================")
        print("請在開啟的瀏覽器視窗中完成 Instagram 帳號登入。")
        print("登入成功並看到 IG 首頁或個人主頁後，回到此終端機按 [Enter] 繼續...")
        print("========================================================\n")

        input("完成登入後按 Enter 儲存狀態...")

        context.storage_state(path=STATE_FILE)
        print(f"[+] 登入狀態已成功儲存至：{STATE_FILE}")
        browser.close()


# ============================================================
# 主爬蟲邏輯
# ============================================================
def crawl_instagram_profile():
    captured_posts = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,  # 設為 False 便於觀察與通過基礎反爬
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars"
            ]
        )

        context = browser.new_context(
            storage_state=STATE_FILE,
            viewport={"width": 1280, "height": 900},
            locale="zh-TW",
            timezone_id="Asia/Taipei",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )

        page = context.new_page()

        # 監聽 Network Response 攔截 GraphQL / Feed API
        def handle_response(response):
            url = response.url.lower()
            content_type = response.headers.get("content-type", "").lower()

            # 辨識 IG 貼文載入的 API 標徵
            if any(k in url for k in ["graphql", "query", "feed", "user"]):
                if "json" in content_type or "javascript" in content_type:
                    try:
                        data = response.json()
                        found = extract_posts_from_json(data)
                        if found:
                            captured_posts.extend(found)
                    except Exception:
                        pass

        page.on("response", handle_response)

        print(f"[*] 正在前往目標頁面：{TARGET_URL}")
        page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(5000)

        # 處理常見彈窗（例如：開啟通知、儲存資訊）
        dismiss_buttons = ["稍後再說", "Not Now", "取消", "Cancel"]
        for btn_text in dismiss_buttons:
            try:
                locator = page.locator(f"button:has-text('{btn_text}')")
                if locator.count() > 0:
                    locator.first.click(timeout=2000)
                    print(f"[*] 已關閉彈窗：{btn_text}")
            except Exception:
                pass

        # 檢測是否有登入牆中斷
        if "accounts/login" in page.url:
            print("\n[!] 登入狀態已過期，請重新登入...")
            input("登入完成後按 Enter 繼續...")
            context.storage_state(path=STATE_FILE)

        print(f"[*] 開始模擬滑動頁面（共 {SCROLL_TIMES} 次）以觸發貼文載入...")
        for i in range(SCROLL_TIMES):
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(int(WAIT_SECONDS * 1000))
            print(f"    - 完成第 {i + 1}/{SCROLL_TIMES} 次滾動，目前已累積捕捉 {len(captured_posts)} 筆資料片段")

        # 備援機制：如果網路封包沒有捕捉到足夠貼文，走訪頁面上的貼文彈窗（Modal）
        if len(captured_posts) < 5:
            print("[*] 啟動 DOM 點擊遍歷備援機制...")
            post_links = page.locator('a[href^="/p/"], a[href^="/reel/"]')
            count = post_links.count()
            if count > 0:
                print(f"[*] 畫面偵測到 {count} 篇貼文，依序開啟讀取...")
                post_links.first.click()
                page.wait_for_timeout(3000)

                for _ in range(min(count, 30)):
                    try:
                        # IG 燈箱中的貼文內文通常置於 h1 內
                        caption_el = page.locator('div[role="dialog"] h1')
                        if caption_el.count() > 0:
                            raw_text = caption_el.first.inner_text()
                            text_clean = clean_post_text(raw_text)
                            current_url = page.url
                            captured_posts.append({
                                "shortcode": current_url.strip("/").split("/")[-1],
                                "url": current_url,
                                "date": "",
                                "text": text_clean
                            })

                        # 點擊「下一篇」箭頭按鈕
                        next_btn = page.locator('div[role="dialog"] button:has(svg[aria-label="下一步"]), div[role="dialog"] button:has(svg[aria-label="Next"])')
                        if next_btn.count() > 0:
                            next_btn.first.click()
                            page.wait_for_timeout(2000)
                        else:
                            break
                    except Exception:
                        break

        browser.close()

    # ============================================================
    # 資料去重整理
    # ============================================================
    print("\n[*] 正在清洗與去除重複貼文...")
    unique_posts = []
    seen_texts = set()

    for item in captured_posts:
        text = item["text"]
        if not text or len(text) < 10:
            continue

        # 建立去重特徵（忽略空白後的前 50 個字）
        sim_key = re.sub(r"\s+", "", text)[:50]
        if sim_key in seen_texts:
            continue

        seen_texts.add(sim_key)
        unique_posts.append(item)

    return unique_posts


# ============================================================
# 執行與檔案儲存
# ============================================================
if __name__ == "__main__":
    ensure_login_state()
    posts = crawl_instagram_profile()

    print(f"\n[+] 爬取完成！成功抽取 {len(posts)} 篇不重複貼文內文\n")

    if not posts:
        print("[!] 未抓取到貼文，請確認帳號登入狀態是否正常。")
        exit()

    # 1. 儲存 TXT 檔
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for idx, post in enumerate(posts, 1):
            f.write(f"【第 {idx} 篇】\n")
            if post["url"]:
                f.write(f"貼文網址：{post['url']}\n")
            if post["date"]:
                f.write(f"發布時間：{post['date']}\n")
            f.write("內文：\n")
            f.write(post["text"])
            f.write("\n" + "=" * 60 + "\n\n")

    # 2. 儲存 CSV 檔
    df = pd.DataFrame({
        "id": range(1, len(posts) + 1),
        "post_url": [p["url"] for p in posts],
        "publish_date": [p["date"] for p in posts],
        "caption": [p["text"] for p in posts]
    })
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[+] TXT 檔案已儲存：{os.path.abspath(OUTPUT_TXT)}")
    print(f"[+] CSV 檔案已儲存：{os.path.abspath(OUTPUT_CSV)}")

    # 預覽前 2 筆抓取結果
    print("\n" + "=" * 30 + " 最新貼文預覽 " + "=" * 30)
    for p in posts[:2]:
        print(f"網址: {p['url']}")
        print(f"時間: {p['date']}")
        print("內文前 120 字：")
        print(p["text"][:120] + ("..." if len(p["text"]) > 120 else ""))
        print("-" * 60)