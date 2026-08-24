import csv
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==================== 1. 初始化強效偽裝瀏覽器 ====================
options = uc.ChromeOptions()
options.add_argument('--start-maximized')
options.add_argument('--disable-gpu')

print("正在啟動強效偽裝瀏覽器 (Chrome 150)...")
driver = uc.Chrome(options=options, version_main=150)

target_url = "https://www.dcard.tw/search?query=%E6%9D%B1%E5%90%B3%E8%B3%87%E7%A7%91"
target_count = 200  # 目標爬取篇數
stale_threshold = 10  # 設定連續卡住幾次就自動判定停止

try:
    print("進入 Dcard 搜尋頁面...")
    driver.get(target_url)
    
    wait = WebDriverWait(driver, 15)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/p/"]')))
    time.sleep(random.uniform(3.0, 5.0))
    
    print("模擬人類操作：重新整理網頁...")
    driver.refresh()
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/p/"]')))
    time.sleep(random.uniform(4.0, 6.0))
    
    # ==================== 2. 動態滾動機制 (含卡住自動偵測) ====================
    post_urls = []
    max_scroll_attempts = 100  # 提高上限安全防線
    scroll_count = 0
    
    # ✨ 用於優化偵測的變數
    last_url_count = 0
    no_new_url_streak = 0
    
    print(f"\n🚀 開始自動滾動頁面，目標蒐集 {target_count} 個文章網址...")
    
    while len(post_urls) < target_count and scroll_count < max_scroll_attempts:
        # 隨機向下滾動不規則像素
        scroll_y = random.randint(800, 1200)
        driver.execute_script(f"window.scrollBy(0, {scroll_y});")
        scroll_count += 1
        
        # 滾動後隨機等待，模仿真人滑手機
        time.sleep(random.uniform(2.5, 4.0))
        
        # 撈取當前頁面所有的連結
        elements = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/p/"]')
        for elem in elements:
            url = elem.get_attribute('href')
            if url and "/p/" in url and url not in post_urls:
                post_urls.append(url)
                
        # 💡 檢查本次滾動後，網址數量有沒有增加
        if len(post_urls) == last_url_count:
            no_new_url_streak += 1
            print(f" 滾動第 {scroll_count} 次 -> 網址數停留在 {len(post_urls)} (已連續卡住 {no_new_url_streak} 次) ⚠️")
            print(f"    [提示] 如果這不是頁面底部，請檢查 Chrome 視窗是否跳出了 Cloudflare 驗證挑戰，請手動點選它！")
        else:
            no_new_url_streak = 0  # 有新網址，重置卡住計數器
            print(f" 滾動第 {scroll_count} 次 -> 目前已蒐集到 {len(post_urls)} / {target_count} 個網址")
            
        last_url_count = len(post_urls)
        
        # 💡 自動斷尾：如果連續卡住達到設定次數，自動退出迴圈
        if no_new_url_streak >= stale_threshold:
            print(f"\n🛑 [自動中斷] 網址數量已連續 {stale_threshold} 次沒有變動！")
            print(f"   程式判定可能「已達網頁底部」或「驗證超時未解」。將直接進行下一步爬取。")
            break
            
        if len(post_urls) >= target_count:
            break
            
    # 截取最終需要的數量
    post_urls = post_urls[:target_count]
    print(f"\n🎯 網址蒐集階段結束！最終共取得 {len(post_urls)} 篇獨立文章網址。")
    
    # ==================== 3. 逐一深入內頁爬取 + 即時寫入 CSV ====================
    csv_filename = 'b_dcard.csv'
    print(f"\n建立檔案 {csv_filename}，準備開始逐篇爬取與即時儲存...\n")
    
    with open(csv_filename, 'w', encoding='utf-8-sig', newline='') as csvfile:
        fieldnames = ['標題', '時間', '內文']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for index, post_url in enumerate(post_urls):
            print(f"[{index+1} / {len(post_urls)}] 模擬真人點入: {post_url}")
            driver.get(post_url)
            
            inner_wait = WebDriverWait(driver, 10)
            
            try:
                # 等待文章主體渲染出來
                inner_wait.until(EC.presence_of_element_located((By.TAG_NAME, 'article')))
                time.sleep(random.uniform(4.0, 6.5))
                
                # A. 標題
                title_elem = driver.find_element(By.TAG_NAME, 'h2')
                title = title_elem.text.strip() if title_elem else "無標題"
                
                # B. 時間
                time_elem = driver.find_element(By.TAG_NAME, 'time')
                post_time = time_elem.get_attribute('title') if time_elem else "無時間"
                if not post_time and time_elem:
                    post_time = time_elem.text.strip()
                    
                # C. 內文
                content_elem = driver.find_element(By.CSS_SELECTOR, 'article div div div span')
                content = content_elem.text.strip() if content_elem else "內文讀取失敗"
                
                # 即時寫入 CSV 檔案
                writer.writerow({
                    '標題': title,
                    '時間': post_time,
                    '內文': content
                })
                csvfile.flush()  # 確保硬碟同步
                
                print(f"   └ 成功儲存：《{title[:12]}...》")
                
            except Exception as e:
                print(f"   └ ❌ 該篇文章解析失敗或遭網頁結構阻擋，跳過。")
                continue
                
            # 每篇之間的喘息時間 (隨機 3 ~ 6 秒)
            time.sleep(random.uniform(3.0, 6.0))

    print(f"\n🎉 爬蟲任務圓滿完成！資料已完整安全的儲存在 {csv_filename}")

finally:
    driver.quit()