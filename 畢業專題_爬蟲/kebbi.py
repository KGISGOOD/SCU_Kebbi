# ==============================================================================
# 🍎 macOS 環境安裝套件指令 (請在 macOS 終端機 Terminal / zsh 執行)：
# 
# 安裝所需相依套件：
#   pip install torch pandas numpy tqdm pydantic faiss-cpu sentence-transformers langchain langchain-core langchain-community langchain-huggingface langchain-text-splitters langchain-groq gradio
# ==============================================================================
import os
import re
import time
import pandas as pd
import numpy as np
import torch
import faiss
import random
from tqdm import tqdm
from pydantic import BaseModel, Field, ConfigDict
import os
import warnings
from dotenv import load_dotenv
# LangChain 模組
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 關閉 Hugging Face Tokenizers 平行處理警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 忽略 LangChain 等套件的棄用警告 (DeprecationWarning)
warnings.filterwarnings("ignore")

# ==========================================
# 1. 文件載入模組 (.txt / .csv)
# ==========================================

def load_txt_file(filepath: str) -> list[Document]:
    """載入 TXT 檔案"""
    filename = os.path.splitext(os.path.basename(filepath))[0]
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return [Document(page_content=content, metadata={"source": filename, "filepath": filepath, "type": "txt"})]
    except Exception as e:
        print(f"❌ 讀取 TXT 失敗 ({filepath}): {e}")
        return []


def load_csv_file(filepath: str) -> list[Document]:
    """載入 CSV 檔案，將每一列轉為結構化 Document"""
    filename = os.path.splitext(os.path.basename(filepath))[0]
    documents = []
    try:
        df = pd.read_csv(filepath)
        for idx, row in df.iterrows():
            row_str = "\n".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            documents.append(Document(
                page_content=row_str,
                metadata={"source": filename, "row_index": idx, "filepath": filepath, "type": "csv"}
            ))
        return documents
    except Exception as e:
        print(f"❌ 讀取 CSV 失敗 ({filepath}): {e}")
        return []


def load_documents_from_directory(directory: str) -> list[Document]:
    """
    遞迴掃描目錄（包含 dcard、txt、訪談 等子資料夾）
    自動排除虛擬環境、Git 與暫存檔
    """
    documents = []
    loaded_files = []
    supported_extensions = ['.txt', '.csv']

    # 排除常見系統與環境資料夾
    ignore_dirs = {
        'sample_data', '.config', '.ipynb_checkpoints',
        'faiss_db_store', 'venv', '.venv', '__pycache__', '.git'
    }

    abs_dir = os.path.abspath(directory)
    print(f"🔍 開始掃描目錄: {abs_dir}")

    for root, dirs, files in os.walk(abs_dir):
        # 過濾忽略資料夾
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
    """
    文件切割策略：
    1. CSV 文件：保持一列一 Chunk，保留結構完整。
    2. TXT 文件：依據標題與標點符號進行語意切割。
    """
    chunks = []

    csv_docs = [doc for doc in documents if doc.metadata.get("type") == "csv"]
    txt_docs = [doc for doc in documents if doc.metadata.get("type") == "txt"]

    # 1. CSV 保留獨立結構
    chunks.extend(csv_docs)
    print(f"📊 CSV 紀錄共 {len(csv_docs)} 筆，保留獨立結構。")

    # 2. TXT 進行語意切割
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
# 2. FAISS 向量庫動態索引建構模組
# ==========================================

def create_embeddings(use_cpu: bool = False) -> HuggingFaceEmbeddings:
    """建立 Embeddings 模型（自動適配 CUDA GPU / Apple Silicon MPS / CPU）"""
    if not use_cpu and torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 啟用 NVIDIA GPU 加速: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    elif not use_cpu and torch.backends.mps.is_available():
        device = "mps"
        print("🚀 啟用 Apple Silicon MPS 加速")
    else:
        device = "cpu"
        print("⚙️ 使用 CPU 運算模式")

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )


def build_adaptive_faiss_vectorstore(chunks: list[Document], embeddings_model: HuggingFaceEmbeddings) -> FAISS:
    """根據向量庫筆數動態選擇 FAISS 檢索索引策略"""
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
# 4. RAG QA Chain 系統建置 (資科系學長 Prompt)
# ==========================================

def load_all_vectorstores(parent_directory: str, embeddings_model: HuggingFaceEmbeddings) -> list[FAISS]:
    """掃描指定資料夾下的所有 FAISS 向量庫"""
    vectorstores = []
    for root, dirs, files in os.walk(parent_directory):
        if "index.faiss" in files and ("index.pkl" in files or "index.pki" in files):
            try:
                vs = FAISS.load_local(root, embeddings_model, allow_dangerous_deserialization=True)
                vectorstores.append(vs)
                print(f"✅ 成功載入向量庫: {root}")
            except Exception as e:
                print(f"❌ 載入向量庫失敗 ({root}): {e}")
    return vectorstores


load_dotenv()

def setup_qa_chain(vectorstores: list[FAISS]):
    """設定 Conversational RAG QA Chain (從 .env 讀取 API Key)"""
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

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = """你現在是「東吳大學資料科學系（資科系）」熱心、專業且親切的「學長」。請根據下方檢索到的參考資料與對話歷史，以學長的口吻回答學弟妹（使用者）的問題。

對話歷史：
{chat_history}

參考資料：
{context}

使用者問題：
{question}

回答指南：
1. **依據資料回答**：優先且嚴格根據「參考資料」內容來回答。如果資料庫裡沒有足夠的資訊，請親切且誠實地告知：「拍謝，學長手邊的資料庫裡目前沒有足夠的資訊可以回答這個問題喔！」
2. **角色與語氣**：保持東吳資科系學長熱心、條理分明且平易近人的風格，使用繁體中文回答。
3. **校系比較立場**：當學弟妹詢問東吳資科與其他校系比較時，請充分展現東吳資科的特色與優勢（如扎實課程、實務資源與系友網絡），表達對系上的肯定；但請保持學長客觀分享的態度，切勿出現「建議你選哪間」或「哪間比較好」等硬性下結論的說辭。
4. **回答方式**:用聊天口語敘述的方式呈現，字數不超過100字且不要是條列式呈現！！！

請以學長的身份開始回答，並且開頭不用自我介紹，直接針對問題用敘述方式回答：
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )
    return qa_chain, retriever


# ==========================================
# 5. 主程式 Pipeline 與 終端機互動
# ==========================================

def main_build_and_launch():
    # 取得當前腳本所在的目錄絕對路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    DB_SAVE_DIR = os.path.join(BASE_DIR, "faiss_db_store")

    # 1. 建立 Embedding 模型
    embeddings_model = create_embeddings(use_cpu=False)

    # 2. 檢查向量庫是否存在，不存在則掃描目錄（自動讀取 dcard、txt、訪談 等子資料夾）
    if not os.path.exists(DB_SAVE_DIR):
        print("\n🔨 向量庫不存在，開始掃描本機目錄下的檔案...")

        raw_documents = load_documents_from_directory(DATA_DIR)

        if raw_documents:
            chunks = split_documents(raw_documents)
            vectorstore = build_adaptive_faiss_vectorstore(chunks, embeddings_model)
            vectorstore.save_local(DB_SAVE_DIR)
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

    while True:
        try:
            user_input = input("學弟妹：").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n學長：有任何問題隨時再來問我，加油啦！掰掰！👋")
                break

            start_time = time.perf_counter()

            response = qa_chain.invoke({"question": user_input})
            answer = response.get('answer', '')

            elapsed_time = time.perf_counter() - start_time

            print(f"\n學長：{answer}")
            # print(f"⏱️ [回答耗時：{elapsed_time:.2f} 秒]\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\n學長：對話已被中斷，掰掰！👋")
            break
        except Exception as e:
            print(f"\n❌ 發生錯誤: {str(e)}\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main_build_and_launch()