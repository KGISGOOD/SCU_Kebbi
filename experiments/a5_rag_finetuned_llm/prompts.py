"""Minimal A0 baseline_basic_v1 content port, using native chat roles instead.

Importing A0.prompts imports A0.config, whose Settings creates directories.
Keep this compatibility layer local so A5 never initializes A0's writable config.
No manual Llama control tokens and no wrapping of A0's serialized build_prompt.
"""

PROMPT_VERSION = "a5_a0_baseline_basic_v1_native_chat_v1"
SYSTEM_PROMPT = """你是「資科萬事通」，專門回答東吳大學巨量資料管理學院資料科學系的相關問題。
請主要根據提供的參考資料回答，使用繁體中文，內容清楚、簡潔。
如果參考資料不足以確認答案，請明確說明目前資料不足，不要自行猜測或編造。
涉及日期、招生、學分或規定時，不要把不同年度的資訊混為一談；無法確認目前最新資訊時，應說明無法確認。
如果問題與東吳資料科學系完全無關，請簡短說明本系統主要提供系所相關資訊。
測試參考日期為 2026 年 10 月 19 日；這是固定的離線測試基準，不代表資料一定是最新的。"""


def format_documents(documents):
    """Same content/order/separators as A0.format_documents; never re-sort."""
    if not documents:
        return "無相關資訊"
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source_file", "未知來源")
        content_type = doc.metadata.get("content_type", "未知類型")
        parts.append("[資料 %d] 來源：%s | 類型：%s\n%s" %
                     (i, source, content_type, doc.page_content))
    return "\n\n---\n\n".join(parts)


def build_messages(query, documents):
    context = format_documents(documents)
    user = "參考資料：\n\n%s\n\n問題：\n\n%s\n\n請回答：" % (context, query)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
