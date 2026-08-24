import csv
import time
import random
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==================== 1. 初始化強效偽裝瀏覽器 ====================
options = uc.ChromeOptions()

# 啟用常規真人視窗大小，避免預設的測試視窗尺寸暴露特徵
options.add_argument('--start-maximized')
options.add_argument('--disable-gpu')

print("正在啟動強效偽裝瀏覽器 (undetected-chromedriver)...")

# 💡 關鍵修正：加上 version_main=150，強制對齊你電腦的 Chrome 版本
driver = uc.Chrome(options=options, version_main=150)

target_url = "https://www.dcard.tw/search?query=%E6%9D%B1%E5%90%B3%E8%B3%87%E7%A7%91"
scraped_data = []

try:
    print("進入 Dcard 搜尋頁面...")
    driver.get(target_url)
    
    # 建立主視窗的等待物件 (最多等待 15 秒)
    wait = WebDriverWait(driver, 15)
    
    # 等待搜尋列表的文章區塊渲染出來
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/p/"]')))
    time.sleep(random.uniform(3.0, 5.0))
    
    # 💡 模擬人類行為：剛進入時手動重新整理網頁
    print("模擬人類操作：重新整理網頁...")
    driver.refresh()
    
    # 重新整理後，等待元素再次出現
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/p/"]')))
    time.sleep(random.uniform(4.0, 6.0))
    
    # ==================== 2. 模擬真人分段滾動頁面 ====================
    print("開始模擬人類不規則滾動頁面，加載更多文章...")
    for i in range(3):  # 滾動 3 次獲取更多基礎資料
        scroll_y = random.randint(700, 1000)
        driver.execute_script(f"window.scrollBy(0, {scroll_y});")
        print(f" 頁面不規則向下滾動 {scroll_y} 像素...")
        time.sleep(random.uniform(3.0, 5.0))  # 模仿人類停頓閱讀
    
    # ==================== 3. 提取搜尋列表網址並去重 ====================
    print("\n正在解析文章列表...")
    post_elements = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/p/"]')
    
    post_urls = []
    for elem in post_elements:
        url = elem.get_attribute('href')
        # 確保網址存在、是完整的 Dcard 文章連結，且不重複加入
        if url and "/p/" in url and url not in post_urls:
            post_urls.append(url)
            
    print(f"👍 成功偵測到 {len(post_urls)} 篇獨特的文章網址。")
    
    # ==================== 4. 逐一深入內頁爬取 ====================
    # 限制爬取前 5 篇作為範例，避免短時間內狂點被系統判定為異常流量
    max_scrape_count = min(5, len(post_urls))
    
    for index, post_url in enumerate(post_urls[:max_scrape_count]):
        print(f"\n[{index+1}/{max_scrape_count}] 模擬真人點入網址: {post_url}")
        driver.get(post_url)
        
        # 建立內頁專用的等待區區塊
        inner_wait = WebDriverWait(driver, 10)
        
        try:
            # 確保內頁的 article 核心區塊已加載
            inner_wait.until(EC.presence_of_element_located((By.TAG_NAME, 'article')))
            time.sleep(random.uniform(4.0, 6.0))  # 真人閱讀緩衝
            
            # A. 爬取標題 (定位主標題 h2)
            title_elem = driver.find_element(By.TAG_NAME, 'h2')
            title = title_elem.text.strip() if title_elem else "無標題"
            
            # B. 爬取時間
            time_elem = driver.find_element(By.TAG_NAME, 'time')
            post_time = time_elem.get_attribute('title') if time_elem else "無時間"
            if not post_time and time_elem:
                post_time = time_elem.text.strip()
                
            # C. 爬取內文
            content_elem = driver.find_element(By.CSS_SELECTOR, 'article div div div span')
            content = content_elem.text.strip() if content_elem else "內文讀取失敗"
            
            print(f" ✅ 成功爬取 -> 《{title[:12]}...》")
            
            scraped_data.append({
                '標題': title,
                '時間': post_time,
                '內文': content
            })
            
        except Exception as e:
            print(f"❌ 該篇文章部分欄位解析跳過")
            continue
            
        # 離開文章、準備前往下一篇前的真人喘息時間
        time.sleep(random.uniform(3.5, 6.0))

    # ==================== 5. 儲存成 CSV ====================
    csv_filename = 'a_dcard.csv'
    print(f"\n正在將數據寫入 {csv_filename} ...")
    
    with open(csv_filename, 'w', encoding='utf-8-sig', newline='') as csvfile:
        fieldnames = ['標題', '時間', '內文']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in scraped_data:
            writer.writerow(data)
            
    print(f"🎉 畢業專案爬蟲任務完美完成！資料已成功存入 {csv_filename}")

finally:
    # 確保安全關閉瀏覽器
    driver.quit()