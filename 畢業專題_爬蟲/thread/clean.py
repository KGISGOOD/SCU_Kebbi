import pandas as pd

# 1. 定義你指定的關鍵字清單
keywords = ["東吳", "資料科學", "資料科學系", "資科", "東吳資科"]

# 將關鍵字用 "|" (或) 串接成正則表達式
# 其實「東吳|資料科學|資科」就能涵蓋全部，但這裡保險起見完全依照你的清單去比對
pattern = "|".join(keywords)

def clean_thread_data(file_path, output_path):
    try:
        # 讀取 CSV 檔案
        df = pd.read_csv(file_path)
        
        # 檢查是否存在 'text' 欄位
        if 'text' not in df.columns:
            print(f"錯誤：{file_path} 中找不到 'text' 欄位，請檢查欄位名稱。")
            return
            
        # 記錄原本的資料筆數
        original_count = len(df)
        
        # 進行過濾：只保留 text 欄位包含關鍵字的資料
        # astype(str) 確保數值或空值不會導致報錯
        # na=False 代表如果該行是空白 (NaN)，就直接刪除
        cleaned_df = df[df['text'].astype(str).str.contains(pattern, na=False)]
        
        # 儲存清理後的結果 (使用 utf-8-sig 確保 Excel 打開不會亂碼)
        cleaned_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"✨ 檔案【{file_path}】清理完成！")
        print(f"   原始資料：{original_count} 筆 -> 清理後剩餘：{len(cleaned_df)} 筆\n")
        
    except Exception as e:
        print(f"處理檔案 {file_path} 時發生錯誤: {e}")

# 2. 批量處理你上傳的兩個檔案
clean_thread_data('東吳資料科學系_threads_texts.csv', 'cleaned_東吳資料科學系.csv')
clean_thread_data('東吳資科_threads_texts.csv', 'cleaned_東吳資科.csv')