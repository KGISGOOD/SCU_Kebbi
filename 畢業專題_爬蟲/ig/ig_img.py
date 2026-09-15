import os
import re
import json
import time
import html as ihtml
from datetime import datetime
import urllib.request
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

# 圖片儲存設定
DOWNLOAD_IMAGES = True            # 是否自動下載圖片檔案至本地
IMAGE_DIR = "東吳資科_IG圖片"      # 圖片儲存資料夾名稱

# 輸出檔案名稱
OUTPUT_CSV = "東吳資科_IG貼文.csv"
OUTPUT_TXT = "東吳資科_IG貼文.txt"


# ============================================================
# 工具函式：文字清理與圖片下載
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


def download_image(url, save_path):
    """
    下載單張圖片並儲存至指定路徑。
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.instagram.com/"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(save_path, "wb") as f:
                f.write(response.read())
        return True
    except Exception:
        return False


def extract_images_from_node(node):
    """
    從貼文節點中抽取圖片網址（支援多圖輪播與單圖）。
    """
    images = []

    # 1. 處理輪播貼文 (GraphQL: edge_sidecar_to_children)
    if "edge_sidecar_to_children" in node and isinstance(node["edge_sidecar_to_children"], dict):
        edges = node["edge_sidecar_to_children"].get("edges", [])
        for edge in edges:
            c_node = edge.get("node", {})
            d_url = c_node.get("display_url")
            if d_url:
                images.append(d_url)

    # 2. 處理輪播貼文 (Mobile / Feed API: carousel_media)
    elif "carousel_media" in node and isinstance(node["carousel_media"], list):
        for item in node["carousel_media"]:
            candidates = item.get("image_versions2", {}).get("candidates", [])
            if candidates and isinstance(candidates, list):
                images.append(candidates[0].get("url"))
            elif item.get("display_url"):
                images.append(item.get("display_url"))

    # 3. 單圖貼文（或非輪播的頂層預覽圖）
    if not images:
        if "image_versions2" in node and isinstance(node["image_versions2"], dict):
            candidates = node["image_versions2"].get("candidates", [])
            if candidates and isinstance(candidates, list):
                images.append(candidates[0].get("url"))
        elif node.get("display_url"):
            images.append(node.get("display_url"))

    # 去除重複網址並保留原順序
    seen = set()
    cleaned_images = []
    for img in images:
        if img and img not in seen:
            seen.add(img)
            cleaned_images.append(img)

    return cleaned_images


# ============================================================
# 從 Instagram API / GraphQL JSON 遞迴提取貼文內文與圖片
# ============================================================
def extract_posts_from_json(obj):
    """
    遞迴走訪 Instagram 前端 API 回傳的 JSON，抽取 shortcode、內文、時間與圖片。
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

            # 檢查是否為貼文節點（具有 shortcode/code 且具有內文或時間戳記）
            has_timestamp = node.get("taken_at_timestamp") or node.get("taken_at")
            if shortcode and (caption_text is not None or has_timestamp):
                raw_time = has_timestamp
                date_str = ""
                if raw_time:
                    try:
                        date_str = datetime.fromtimestamp(int(raw_time)).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        date_str = ""

                cleaned_caption = clean_post_text(caption_text) if caption_text else ""
                image_urls = extract_images_from_node(node)

                # 只要有內文或圖片即視為有效資料
                if cleaned_caption or image_urls:
                    posts.append({
                        "shortcode": shortcode,
                        "url": f"https://www.instagram.com/p/{shortcode}/",
                        "date": date_str,
                        "text": cleaned_caption,
                        "image_urls": image_urls
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
    若本地無 instagram_state.json，自動打開瀏覽器讓使用者手動登入一次並保存狀態。
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
            headless=False,
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

        # 攔截 Network Response
        def handle_response(response):
            url = response.url.lower()
            content_type = response.headers.get("content-type", "").lower()

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

        # 關閉彈窗
        dismiss_buttons = ["稍後再說", "Not Now", "取消", "Cancel"]
        for btn_text in dismiss_buttons:
            try:
                locator = page.locator(f"button:has-text('{btn_text}')")
                if locator.count() > 0:
                    locator.first.click(timeout=2000)
                    print(f"[*] 已關閉彈窗：{btn_text}")
            except Exception:
                pass

        # 檢測登入狀態
        if "accounts/login" in page.url:
            print("\n[!] 登入狀態已過期，請重新登入...")
            input("登入完成後按 Enter 繼續...")
            context.storage_state(path=STATE_FILE)

        print(f"[*] 開始模擬滑動頁面（共 {SCROLL_TIMES} 次）以觸發貼文載入...")
        for i in range(SCROLL_TIMES):
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(int(WAIT_SECONDS * 1000))
            print(f"    - 完成第 {i + 1}/{SCROLL_TIMES} 次滾動，目前已累積捕捉 {len(captured_posts)} 筆資料片段")

        # 備援機制：點擊 DOM 燈箱走訪
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
                        current_url = page.url
                        current_shortcode = current_url.strip("/").split("/")[-1]

                        caption_el = page.locator('div[role="dialog"] h1')
                        raw_text = caption_el.first.inner_text() if caption_el.count() > 0 else ""
                        text_clean = clean_post_text(raw_text)

                        # 抽取燈箱中的圖片網址（排除大頭貼）
                        dialog_imgs = page.locator('div[role="dialog"] article img')
                        if dialog_imgs.count() == 0:
                            dialog_imgs = page.locator('div[role="dialog"] img')

                        dom_img_urls = []
                        for img_idx in range(dialog_imgs.count()):
                            src = dialog_imgs.nth(img_idx).get_attribute("src")
                            alt = dialog_imgs.nth(img_idx).get_attribute("alt") or ""
                            if src and "profile" not in alt.lower() and "個人檔案" not in alt:
                                if src not in dom_img_urls:
                                    dom_img_urls.append(src)

                        captured_posts.append({
                            "shortcode": current_shortcode,
                            "url": current_url,
                            "date": "",
                            "text": text_clean,
                            "image_urls": dom_img_urls
                        })

                        # 下一篇
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
    # 資料去重與圖片清單合併
    # ============================================================
    print("\n[*] 正在清洗與去除重複貼文...")
    posts_by_code = {}
    unique_posts = []

    for item in captured_posts:
        code = item.get("shortcode")
        if not code:
            continue

        if code not in posts_by_code:
            posts_by_code[code] = item
            unique_posts.append(item)
        else:
            # 合併可能在不同封包中抽到的更多圖片網址
            existing_imgs = posts_by_code[code].get("image_urls", [])
            for u in item.get("image_urls", []):
                if u not in existing_imgs:
                    existing_imgs.append(u)
            posts_by_code[code]["image_urls"] = existing_imgs

            if not posts_by_code[code].get("text") and item.get("text"):
                posts_by_code[code]["text"] = item.get("text")

    return unique_posts


# ============================================================
# 執行與檔案儲存
# ============================================================
if __name__ == "__main__":
    ensure_login_state()
    posts = crawl_instagram_profile()

    print(f"\n[+] 爬取完成！成功抽取 {len(posts)} 篇不重複貼文\n")

    if not posts:
        print("[!] 未抓取到貼文，請確認帳號登入狀態是否正常。")
        exit()

    # 下載圖片至本地
    if DOWNLOAD_IMAGES:
        print(f"[*] 開始下載貼文圖片至資料夾：{IMAGE_DIR} ...")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        total_images = sum(len(p.get("image_urls", [])) for p in posts)
        downloaded_count = 0

        for post_idx, post in enumerate(posts, 1):
            local_files = []
            code = post["shortcode"] or f"post_{post_idx}"
            img_list = post.get("image_urls", [])

            for img_idx, img_url in enumerate(img_list, 1):
                filename = f"{post_idx:03d}_{code}_{img_idx}.jpg"
                filepath = os.path.join(IMAGE_DIR, filename)

                if download_image(img_url, filepath):
                    local_files.append(filepath)
                    downloaded_count += 1

                time.sleep(0.15)  # 避免過度頻繁請求觸發防爬

            post["local_files"] = local_files
            if img_list:
                print(f"    - 第 {post_idx}/{len(posts)} 篇 ({code})：完成 {len(local_files)}/{len(img_list)} 張圖片下載")

        print(f"[+] 圖片下載完成！共成功下載 {downloaded_count}/{total_images} 張圖片。\n")
    else:
        for post in posts:
            post["local_files"] = []

    # 儲存 CSV 檔（僅保留 id, publish_date, caption）
    df = pd.DataFrame({
        "id": range(1, len(posts) + 1),
        "publish_date": [p["date"] for p in posts],
        "caption": [p["text"] for p in posts]
    })
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[+] CSV 檔案已儲存：{os.path.abspath(OUTPUT_CSV)}")
    if DOWNLOAD_IMAGES:
        print(f"[+] 圖片資料夾位置：{os.path.abspath(IMAGE_DIR)}")

    # 預覽前 2 筆抓取結果
    print("\n" + "=" * 30 + " 最新貼文預覽 " + "=" * 30)
    for p in posts[:2]:
        print(f"網址: {p['url']}")
        print(f"時間: {p['date']}")
        print(f"圖片張數: {len(p.get('image_urls', []))}")
        if p.get("local_files"):
            print(f"本地檔案: {p['local_files'][0]} 等")
        print("內文前 100 字：")
        print(p["text"][:100] + ("..." if len(p["text"]) > 100 else ""))
        print("-" * 60)