from playwright.sync_api import sync_playwright

STATE_FILE = "threads_state.json"

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=False,
        channel="chrome"
    )

    context = browser.new_context(
        viewport={"width": 1280, "height": 900},
        locale="zh-TW",
        timezone_id="Asia/Taipei"
    )

    page = context.new_page()

    page.goto("https://www.threads.com/", wait_until="domcontentloaded")

    print("請在打開的瀏覽器中手動登入 Threads / Instagram。")
    print("登入完成並確認可以看到 Threads 頁面後，回到終端機按 Enter。")

    input("登入完成後按 Enter 繼續...")

    context.storage_state(path=STATE_FILE)

    print(f"登入狀態已儲存到：{STATE_FILE}")

    browser.close()