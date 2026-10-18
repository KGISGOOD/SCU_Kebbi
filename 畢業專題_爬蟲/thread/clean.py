import re
import pandas as pd

# 檔案路徑設定
INPUT_CSV = "東吳資料科學系_threads_texts.csv"
OUTPUT_CSV = "東吳資料科學系_threads_texts_cleaned.csv"
OUTPUT_TXT = "東吳資料科學系_threads_texts_cleaned.txt"

# 1. 讀取 CSV
df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
original_count = len(df)

# 2. 過濾條件：去除開頭為 "Photo by"（包含忽略大小寫與可能的前導空白）
# regex: ^\s*Photo by[\s\xa0]
is_photo_by = df["text"].astype(str).str.contains(r"^\s*Photo by[\s\xa0]", regex=True, case=False)

# 保留非 Photo by 的資料
df_cleaned = df[~is_photo_by].copy()

# 3. 重新編排流水號 id
df_cleaned["id"] = range(1, len(df_cleaned) + 1)

# 4. 輸出更新後的 CSV（帶 BOM 防 Excel 亂碼）
df_cleaned.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# 5. 同步輸出乾淨的 TXT 檔
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for _, row in df_cleaned.iterrows():
        f.write(f"{row['id']}. {row['text']}\n\n")

print(f"原始筆數：{original_count} 筆")
print(f"刪除筆數：{is_photo_by.sum()} 筆 (Photo by...)")
print(f"剩餘筆數：{len(df_cleaned)} 筆")
print(f"已輸出乾淨檔案：\n- {OUTPUT_CSV}\n- {OUTPUT_TXT}")