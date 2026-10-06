# ==============================================================================
# 📦 Windows 環境安裝套件指令 (請在命令提示字元 CMD 或 PowerShell 執行)：
# pip install torch pandas numpy tqdm pydantic faiss-cpu sentence-transformers langchain-core langchain-community langchain-huggingface langchain-text-splitters langchain-groq
# ==============================================================================

import os
import sys
import time
import random
import pickle
import warnings
import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Windows 終端機 UTF-8 編碼強制支援 (避免 Windows CMD/PowerShell 繁體中文亂碼)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# 關閉 Tokenizers 平行警告與 LangChain 棄用警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# LangChain 模組 (全面使用 langchain_core，完全不依賴 langchain.chains)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# ==========================================
# 1. 文件載入模組 (.txt / .csv，相容 Windows 多種編碼)
# ==========================================

def load_txt_file(filepath: str) -> list[Document]:
    """載入 TXT 檔案（自動嘗試 UTF-8、UTF-8-SIG、CP950 避免 Windows 讀檔報錯）"""
    filename = os.path.splitext(os.path.basename(filepath))[0]
    encodings = ['utf-8', 'utf-8-sig', 'cp950', 'gbk']
    
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            return [Document(page_content=content, metadata={"source": filename, "filepath": filepath, "type": "txt"})]
        except (UnicodeDecodeError, Exception):
            continue

    print(f"❌ 讀取 TXT 失敗 ({filepath}): 無法解析檔案編碼")
    return []


def load_csv_file(filepath: str) -> list[Document]:
    """載入 CSV 檔案（相容 Windows Excel 匯出的 ANSI/CP950 與 UTF-8 編碼）"""
    filename = os.path.splitext(os.path.basename(filepath))[0]
    documents = []
    encodings = ['utf-8', 'utf-8-sig', 'cp950', 'gbk']
    df = None

    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df is not None:
        try:
            for idx, row in df.iterrows():
                row_str = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                documents.append(Document(
                    page_content=row_str,
                    metadata={"source": filename, "row_index": idx, "filepath": filepath, "type": "csv"}
                ))
            return documents
        except Exception as e:
            print(f"❌ 解析 CSV 欄位失敗 ({filepath}): {e}")
            return []
    else:
        print(f"❌ 讀取 CSV 失敗 ({filepath}): 無法以常見編碼讀取")
        return []


def load_documents_from_directory(directory: str) -> list[Document]:
    """
    遞迴掃描目錄（包含 dcard、txt、訪談 等子資料夾）
    自動排除環境資料夾與系統暫存
    """
    documents = []
    loaded_files = []
    supported_extensions = ['.txt', '.csv']

    ignore_dirs = {
        'sample_data', '.config', '.ipynb_checkpoints',
        'faiss_db_store', 'venv', '.venv', '__pycache__', '.git'
    }

    abs_dir = os.path.abspath(directory)
    print(f"🔍 開始掃描目錄: {abs_dir}")

    for root, dirs, files in os.walk(abs_dir):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]

        for filename in files:
            if filename.startswith('.'):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_extensions:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, abs_dir)
                docs = []

                if ext == '.txt':
                    docs = load_txt_file(filepath)
                elif ext == '.csv':
                    docs = load_csv_file(filepath)

                if docs:
                    documents.extend(docs)
                    loaded_files.append(rel_path)
                    print(f"  📥 成功讀取檔案: {rel_path}")

    print("=" * 50)
    print(f"✅ 掃描完成！共成功載入 {len(documents)} 份原始紀錄/文件。")
    if loaded_files:
        print("📋 準備轉換為向量的檔案清單：")
        for idx, f in enumerate(loaded_files, 1):
            print(f"   {idx}. {f}")
    print("=" * 50)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """文件切割策略：CSV 保持單列完整，TXT 依語意段落切割"""
    chunks = []

    csv_docs = [doc for doc in documents if doc.metadata.get("type") == "csv"]
    txt_docs = [doc for doc in documents if doc.metadata.get("type") == "txt"]

    chunks.extend(csv_docs)
    print(f"📊 CSV 紀錄共 {len(csv_docs)} 筆，保留獨立結構。")

    if txt_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=[
                "\n# ", "\n## ", "\n### ",
                "\n\n",
                "\n",
                "。", "；", "！", "？",
                " ", ""
            ]
        )
        txt_chunks = text_splitter.split_documents(txt_docs)
        chunks.extend(txt_chunks)
        print(f"📄 TXT 文件切割完成，共產生 {len(txt_chunks)} 個文本塊 (Chunks)。")

    print(f"✨ 全系統最終整合 Chunk 總數: {len(chunks)} 個")
    return chunks


# ==========================================
# 2. FAISS 向量庫動態索引建構模組 (相容 Windows 中文路徑)
# ==========================================

def safe_save_faiss(vectorstore: FAISS, folder_path: str):
    """自訂安全儲存 FAISS 向量庫（避開 Windows C++ fopen 無法解析中文路徑之問題）"""
    os.makedirs(folder_path, exist_ok=True)
    index_file = os.path.join(folder_path, "index.faiss")
    pkl_file = os.path.join(folder_path, "index.pkl")

    chunk = faiss.serialize_index(vectorstore.index)
    with open(index_file, "wb") as f:
        f.write(chunk.tobytes())

    with open(pkl_file, "wb") as f:
        pickle.dump((vectorstore.docstore, vectorstore.index_to_docstore_id), f)


def safe_load_faiss(folder_path: str, embeddings_model: HuggingFaceEmbeddings) -> FAISS:
    """自訂安全載入 FAISS 向量庫（避開 Windows C++ fopen 無法解析中文路徑之問題）"""
    index_file = os.path.join(folder_path, "index.faiss")
    pkl_file = os.path.join(folder_path, "index.pkl")

    with open(index_file, "rb") as f:
        index_data = f.read()
    chunk = np.frombuffer(index_data, dtype=np.uint8)
    index = faiss.deserialize_index(chunk)

    with open(pkl_file, "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)

    return FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )


def create_embeddings(use_cpu: bool = False) -> HuggingFaceEmbeddings:
    """建立 Embeddings 模型（Windows 支援 NVIDIA CUDA GPU 與 CPU）"""
    if not use_cpu and torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 啟用 NVIDIA GPU 加速: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        print("⚙️ 使用 CPU 運算模式")

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )


def build_adaptive_faiss_vectorstore(chunks: list[Document], embeddings_model: HuggingFaceEmbeddings) -> FAISS:
    """根據向量筆數自動挑選最合適的 FAISS 索引演算法"""
    if not chunks:
        raise ValueError("輸入的 chunks 列表為空！")

    texts = [doc.page_content for doc in chunks]

    print("⏳ 正在計算向量嵌入 (Embedding)...")
    embeddings_list = embeddings_model.embed_documents(texts)
    embeddings_np = np.array(embeddings_list, dtype=np.float32)

    faiss.normalize_L2(embeddings_np)
    n_samples, d = embeddings_np.shape
    print(f"📊 資料庫規模: {n_samples} 筆向量 | 維度: {d}")

    if n_samples < 10000:
        print("⚡ 檢索策略: [Exact Search - IndexFlatIP] (數據量 < 10,000)")
        index = faiss.IndexFlatIP(d)
    elif n_samples < 100000:
        print("⚡ 檢索策略: [Graph Search - IndexHNSWFlat] (10,000 <= 數據量 < 100,000)")
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = 64
    else:
        nlist = int(4 * np.sqrt(n_samples))
        print(f"⚡ 檢索策略: [Inverted File Search - IndexIVFFlat, nlist={nlist}] (數據量 >= 100,000)")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        print("🏋️ 正在訓練 IVF 聚類中心...")
        index.train(embeddings_np)
        index.nprobe = min(16, nlist)

    index.add(embeddings_np)

    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(chunks)})
    index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

    vectorstore = FAISS(
        embedding_function=embeddings_model,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore


# ==========================================
# 3. 跨庫與動態檢索器 (Multi-Vectorstore Retriever)
# ==========================================

class MultiVectorstoreRetriever(BaseRetriever, BaseModel):
    vectorstores: list = Field(default_factory=list)
    top_k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        all_docs = []
        for vs in self.vectorstores:
            if vs is None:
                continue
            docs = vs.similarity_search(query, k=self.top_k)
            all_docs.extend(docs)

        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return unique_docs[:self.top_k]


# ==========================================
# 4. RAG QA Chain 系統建置 (LCEL 架構)
# ==========================================

def load_all_vectorstores(parent_directory: str, embeddings_model: HuggingFaceEmbeddings) -> list[FAISS]:
    """掃描指定資料夾下的所有 FAISS 向量庫"""
    vectorstores = []
    for root, dirs, files in os.walk(parent_directory):
        if "index.faiss" in files and ("index.pkl" in files or "index.pki" in files):
            try:
                vs = safe_load_faiss(root, embeddings_model)
                vectorstores.append(vs)
                print(f"✅ 成功載入向量庫: {root}")
            except Exception as e:
                print(f"❌ 載入向量庫失敗 ({root}): {e}")
    return vectorstores


load_dotenv()


def format_docs(docs: list[Document]) -> str:
    """將檢索到的 Document 轉為純文字 Context"""
    return "\n\n".join(doc.page_content for doc in docs)


def setup_qa_chain(vectorstores: list[FAISS]):
    """設定基於 LCEL 的 Conversational RAG QA Chain"""
    retriever = MultiVectorstoreRetriever(vectorstores=vectorstores, top_k=5)

    # 從環境變數讀取金鑰清單
    api_keys_str = os.getenv("GROQ_API_KEYS", "")
    GROQ_API_KEYS = [k.strip() for k in api_keys_str.split(",") if k.strip()]

    if not GROQ_API_KEYS:
        raise ValueError("❌ 未在 .env 檔案中找到有效的 GROQ_API_KEYS，請確認設定！")

    groq_api_key = random.choice(GROQ_API_KEYS)
    print(f"🎲 成功選用 API Key: {groq_api_key[:12]}...")

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-20b",
        temperature=1.0
    )

    # 1. 歷史對話改寫檢索 Query 的 Prompt
    contextualize_q_system_prompt = (
        "請根據對話歷史與使用者的最新問題，將其改寫為一個不依賴上下文、獨立完整的檢索問題。"
        "不要回答問題，只需改寫，若不需要改寫則原樣返回。"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_query_chain = contextualize_q_prompt | llm | StrOutputParser()

    # 2. 學長角色問答 Prompt
    system_prompt ="""你現在是「東吳大學資料科學系（資科系）」熱心、專業且親切的學長。請根據「參考資料」、「對話歷史」以及使用者問題，判斷使用者真正想了解的事情，並以學長的口吻自然回答。

對話歷史：
{chat_history}

參考資料：
{context}

使用者問題：
{question}

回答規則：

1. 資料優先
優先使用「參考資料」回答問題。若參考資料中有直接對應的資訊，應以資料內容為主要依據，不要自行捏造資料。

2. 關鍵字與意圖判斷
如果參考資料沒有直接回答使用者問題，但資料中存在與問題「主題、關鍵字或概念」相關的資訊，請不要直接拒答。
請先判斷使用者可能真正想了解的意圖，再利用相關資料提供「最接近且有幫助」的回答。

例如：
使用者問：「資科系有哪些課可以選？」
若資料沒有完整列出所有選修課，但參考資料提到「計算機概論、資料分析、人工智慧」等課程，請根據這些相關資訊回答目前能確認的內容，而不是直接說資料不足。

3. 相關資訊延伸
當資料不足以完整回答問題時，可以從參考資料中找出「相關主題」進行有限度延伸。
例如使用者問「資科系數學會不會很難？」
若資料沒有直接說明難度，但資料提到線性代數、微積分、統計等課程，可以回答：「系上確實會接觸線性代數、統計等數學相關課程，所以有一定數學基礎會比較好適應。」
但不要自行捏造課程內容、學分、難度或制度。

4. 推測與事實必須區分
如果答案包含根據相關資料做出的合理推論，請使用「看起來」、「如果你是想問……」、「從系上的課程來看……」等自然語氣，讓使用者知道這是根據相關資訊做出的判斷，而不是資料庫明確記載的事實。

禁止把推測內容描述成「系上一定如此」或「資料明確指出」。

5. 完全無關時才拒答
只有在參考資料與使用者問題完全沒有相關資訊，也無法透過關鍵字或主題判斷出合理意圖時，才表示資料不足。

此時不要使用固定、機械式的拒答句，而要自然地說明目前能確認的範圍，例如：
「這部分學長目前查到的資料比較少，不太敢直接亂講。如果你是想問資科系的課程、學習內容或畢業方向，我可以先跟你介紹目前資料裡有提到的部分。」

6. 回答風格
使用繁體中文，以「東吳資科系學長」的聊天口吻回答。
語氣自然、親切、像學長與學弟妹聊天，不要像客服或百科全書。
不要自我介紹，直接回答問題。
不要使用條列式。
不要過度正式，也不要使用過多專業術語。
不要為了湊字數而重複內容。

7. 校系比較
當使用者詢問東吳資科與其他校系比較時，可以客觀分享東吳資科的課程、實務資源與系上特色，但不要直接替使用者決定哪間比較好，也不要出現「你應該選東吳」之類的強制性結論。

8. 回答長度
一般問題控制在約 50～80 字。
如果問題需要較多背景才能說清楚，可以適度增加至約 100 字，但仍以簡潔、自然為原則。

9. 最重要的回答決策流程
回答前依序判斷：

① 參考資料是否有直接答案？
→ 有：直接根據資料回答。

② 沒有直接答案，但是否有相關關鍵字、課程、主題或概念？
→ 有：判斷使用者意圖，利用相關資料回答最接近的內容，必要時明確表示這是根據相關資訊推論。

③ 完全沒有相關資料？
→ 不要硬掰；自然說明目前資料不足，並盡可能指出可以回答的相關方向。

核心原則：
「不要因為沒有完全匹配的資料就立刻拒答；先找相關資訊、判斷使用者意圖，再提供最有幫助且不超出資料可信範圍的回答。」
"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # 動態判斷是否需要經過歷史改寫並執行檢索
    def retrieve_context(input_data: dict) -> str:
        if input_data.get("chat_history"):
            query = history_query_chain.invoke(input_data)
        else:
            query = input_data["input"]
        docs = retriever.invoke(query)
        return format_docs(docs)

    # 組合 LCEL Pipeline
    rag_chain = (
        RunnablePassthrough.assign(context=RunnableLambda(retrieve_context))
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ==========================================
# 5. 主程式 Pipeline 與 終端機互動
# ==========================================

def main_build_and_launch():
    # 取得目前腳本所在的目錄絕對路徑（跨平台相容 Windows / Mac）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    DB_SAVE_DIR = os.path.join(BASE_DIR, "faiss_db_store")

    # 1. 建立 Embedding 模型
    embeddings_model = create_embeddings(use_cpu=False)

    # 2. 檢查向量庫檔案是否完整存在，若任一檔案遺失則重新掃描建庫
    index_file = os.path.join(DB_SAVE_DIR, "index.faiss")
    pkl_file = os.path.join(DB_SAVE_DIR, "index.pkl")

    if not (os.path.exists(index_file) and os.path.exists(pkl_file)):
        print("\n🔨 向量庫索引檔案不存在或不完整，開始掃描本機目錄下的檔案...")

        raw_documents = load_documents_from_directory(DATA_DIR)

        if raw_documents:
            chunks = split_documents(raw_documents)
            vectorstore = build_adaptive_faiss_vectorstore(chunks, embeddings_model)
            safe_save_faiss(vectorstore, DB_SAVE_DIR)
            print(f"💾 向量庫已成功儲存至: {DB_SAVE_DIR}")
        else:
            print("\n⚠️ 未在目錄中找到任何 `.txt` 或 `.csv` 檔案！")
            print("💡 請確認 dcard、txt、訪談 資料夾內有放入文字檔案。")
            return

    # 3. 載入向量庫
    vectorstores = load_all_vectorstores(DB_SAVE_DIR, embeddings_model)
    if not vectorstores:
        print("❌ 未能成功載入任何向量庫。")
        return

    # 4. 初始化 QA Chain
    qa_chain, retriever = setup_qa_chain(vectorstores)

    # 5. 終端機互動對話介面
    print("\n" + "=" * 50)
    print("🎓 東吳資科系學長 - TXT / CSV 智慧 RAG 問答助手 (終端機模式)")
    print("學弟妹好！把檔案相關的問題丟過來，學長幫你從資料庫找出答案！")
    print("💡 提示：輸入 'exit'、'quit' 或 'q' 即可結束對話。")
    print("=" * 50 + "\n")

    chat_history = []

    while True:
        try:
            user_input = input("學弟妹：").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n學長：有任何問題隨時再來問我，加油啦！掰掰！👋")
                break

            start_time = time.perf_counter()

            # LCEL invoke 直接輸出模型回應字串
            answer = qa_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            # 更新對話紀錄
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

            elapsed_time = time.perf_counter() - start_time

            print(f"\n學長：{answer}")
            print(f"⏱️ [回答耗時：{elapsed_time:.2f} 秒]\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\n學長：對話已被中斷，掰掰！👋")
            break
        except Exception as e:
            print(f"\n❌ 發生錯誤: {str(e)}\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main_build_and_launch()