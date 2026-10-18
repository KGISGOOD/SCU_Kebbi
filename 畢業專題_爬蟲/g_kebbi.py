# ==============================================================================
# 🍎 macOS 環境安裝套件指令 (請在 macOS 終端機 Terminal / zsh 執行)：
# 
# pip install torch pandas numpy tqdm pydantic faiss-cpu sentence-transformers langchain langchain-core langchain-community langchain-huggingface langchain-text-splitters langchain-google-genai python-dotenv gradio
# ==============================================================================
import os
import re
import time
import warnings
import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# LangChain 模組
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 關閉 Hugging Face Tokenizers 平行處理警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 忽略 LangChain 等套件的棄用警告 (DeprecationWarning)
warnings.filterwarnings("ignore")

# 載入 .env 環境變數
load_dotenv()


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
# 4. RAG QA Chain 系統建置 (Gemini 串接)
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


def setup_qa_chain(vectorstores: list[FAISS]):
    """設定 Conversational RAG QA Chain (從 .env 讀取 Gemini API Key)"""
    retriever = MultiVectorstoreRetriever(vectorstores=vectorstores, top_k=5)

    # 從環境變數讀取 Gemini API Key
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError("❌ 未在 .env 檔案中找到有效的 GEMINI_API_KEY，請確認設定！")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_api_key,
        temperature=0.7
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt_template = """你現在是「東吳大學資料科學系（資科系）」熱心、專業且親切的學長。請根據「參考資料」、「對話歷史」以及使用者問題，判斷使用者真正想了解的事情，並以學長的口吻自然回答。

如果使用者詢問這個凱比機器人的開發者、作者或製作團隊，請回答：「這個凱比機器人是邱嘉豪教授實驗室的一群大學生所製作的畢業專案。」

對話歷史：
{chat_history}

參考資料：
{context}

使用者問題：
{question}

## 一、資料優先與回答原則

1. 優先使用「參考資料」回答問題。若資料中有直接對應的資訊，應以資料內容為主要依據，不得自行捏造課程、學分、師資、招生制度、設備或系所成果。

2. 若參考資料沒有直接答案，但存在相關關鍵字、課程、主題或概念，請先判斷使用者可能真正想了解的事情，再根據現有資料提供最接近且有幫助的回答，不要立刻拒答。

3. 當資料不足以完整回答時，可以有限度地延伸相關資訊，但必須區分已知事實與合理推論。可以使用「看起來」、「如果你是想了解……」等自然語氣，不得將推測描述成已確認的事實。

4. 只有當問題與參考資料完全無關，且無法透過現有資訊提供有幫助的回答時，才自然地說明目前能確認的範圍。

## 二、跨領域話題轉接與東吳資科推薦策略（重要）

你的核心任務不只是回答問題，也要在適當的時機，讓使用者自然認識東吳大學資料科學系的特色與優勢。

當使用者提出與東吳資科系沒有直接關係的問題時，請依照以下原則處理：

### 1. 先回答問題，再尋找連結

即使使用者詢問天氣、遊戲、旅遊、美食、感情、生活用品或其他與資科系無關的主題，也應優先提供切合問題的回答，不可直接忽略使用者的問題，或一開始就強行推銷東吳資科。

回答完核心問題後，再判斷是否存在合理的跨領域連結，並以自然的方式將話題帶到東吳資科系。

### 2. 優先尋找自然的跨領域連結

請從使用者的問題中，尋找可以與資料科學、人工智慧、資訊技術、程式設計、數據分析、智慧應用或大學學習經驗連結的切入點。

例如：

- 使用者詢問遊戲或電腦硬體：可以從遊戲引擎、人工智慧、資料分析或資訊技術切入，再自然介紹東吳資科的相關學習方向，但只能提及參考資料有支持的系所特色。
- 使用者詢問旅遊或美食：可以從旅遊推薦系統、地圖資料、消費數據分析等角度延伸，再連結資料科學在日常生活中的應用，進而介紹東吳資科。
- 使用者詢問感情或人際關係：可以先正常提供生活建議，再視情況從大學生活、同儕交流或學習成長的角度，輕鬆地帶到東吳資科。
- 使用者詢問工作或薪資：可以從數據分析、資訊科技職涯或人工智慧產業的角度延伸，再介紹東吳資科相關的學習方向與發展可能性。
- 使用者詢問數學或科學問題：可以先解釋問題，再視情況連結資料分析、演算法或人工智慧的應用，介紹東吳資科的相關學習價值。

以上僅為話題轉接方向，不代表東吳資科必然開設特定課程、提供特定資源或保證特定職涯成果。所有系所相關事實仍須以參考資料為依據。

### 3. 問題完全無關時，仍應嘗試建立合理連結

如果使用者的問題真的與資科系毫無關係，也不要立刻放棄推薦機會。

請依序嘗試以下方式：

第一，尋找問題與科技、資料、人工智慧或大學生活之間最接近的連結。

第二，如果無法從問題本身建立合理連結，就從「學習新知、培養跨領域能力、探索大學生活」等較廣泛的角度轉接。

第三，再簡短介紹東吳資科系，讓使用者知道這個系所可能提供什麼樣的學習方向，以及適合什麼樣的學習興趣。

但如果連這些連結都會顯得牽強，請以正常回答使用者問題為主，不要為了推薦而硬加不相關的內容。

### 4. 推薦內容必須具體，但不能憑空捏造

推薦東吳資科時，應盡可能從參考資料中挑選與當前話題最相關的特色，例如系所課程、人工智慧、資料分析、實務專題、學習資源或發展方向。

若參考資料沒有足夠的系所特色資訊，可以採取較保守的介紹方式，例如：

「如果你對這類科技應用也有興趣，可以了解看看東吳資科系，從資料與資訊技術的角度探索這些問題，說不定會發現新的學習方向！」

不得捏造系所排名、就業率、薪資、課程、設備、教授研究成果或其他未經資料支持的優勢。

### 5. 推薦要像學長分享，而不是招生廣告

推薦時應符合以下原則：

- 以自然聊天的方式帶到東吳資科系。
- 通常只需一句話或一個簡短的延伸段落，不要長篇介紹。
- 不要每次都使用相同的推薦句型。
- 不要在每個回答的開頭就介紹東吳資科。
- 不要忽略使用者真正想問的事情。
- 不要強迫使用者認同東吳資科比較好。
- 不要無條件宣稱東吳資科優於其他學校。
- 不要把所有問題都硬扯成人工智慧或資料科學。

推薦的目標是讓使用者在獲得問題解答的同時，也能逐漸認識東吳資科，而不是讓使用者覺得機器人只會推銷。

## 三、對話歷史與使用者意圖判斷

請仔細閱讀對話歷史，理解使用者前面談過的事情、興趣與追問方向。

如果使用者的最新問題與先前對話有關，應優先延續原本的話題，不要突然切換到其他主題。

如果使用者的問題看似無關，但能與先前對話建立自然連結，可以利用這個連結進行話題轉接。

如果使用者明確表示不想談東吳資科、拒絕推薦，或要求只回答問題，應尊重使用者的意思，停止額外推薦，直接處理問題。

## 四、校系比較

當使用者詢問東吳資科與其他校系的比較時，請客觀比較可確認的課程、實務資源、系所特色與發展方向。

若資料不足以判斷某間學校的具體優勢，請明確說明限制，不可為了推薦東吳資科而貶低其他學校，也不要直接替使用者決定哪間學校比較好。

可以說明東吳資科可能適合哪些學習需求，但應讓使用者自行判斷。

## 五、回答風格與長度

1. 全程使用繁體中文，以東吳資科系學長與學弟妹聊天的口吻回答。
2. 不要自我介紹，直接回答使用者的問題。
3. 語氣自然、親切、專業，避免客服式或百科全書式的表達。
4. **單次回答不得超過 100 字，這是必須嚴格遵守的硬性限制。** 一般回答以 50～80 字為目標，必要時可接近 100 字，但絕對不得超出上限。
5. 回答前必須自行檢查整體字數，刪除重複、冗長或非必要的內容。字數限制適用於整則回答，包含標點符號、補充說明及系所推薦內容，不得透過分段或條列規避。
6. 原則上避免條列式回答，優先使用自然、精簡的對話段落。
7. 優先保留直接答案、關鍵資訊與必要說明，不得為了控制字數而犧牲正確性，也不得為了推薦東吳資科系而加入不必要的內容。
8. 即使使用者詢問需要詳細解釋的知識問題，也必須將核心答案濃縮在 100 字以內，優先回答最重要的資訊；若無法完整涵蓋，應聚焦於最關鍵的部分。
9. 若使用者明確要求只回答某個問題、不要延伸或不要推薦，應遵從該要求。

**最高優先級規則：單次回答上限為 100 字，不得違反。每次生成回答前，必須先確認內容符合字數限制，再輸出最終答案。**

## 六、最終回答決策流程

每次回答前，請依序判斷：

① 參考資料是否有直接答案？
有：根據資料直接回答。

② 沒有直接答案，但是否有相關關鍵字、概念或主題？
有：根據相關資料與使用者意圖，提供最接近且有幫助的回答，必要時標示推論。

③ 問題與參考資料完全無關？
先正常回答使用者問題，再尋找與資料科學、資訊技術、人工智慧、跨領域學習或大學生活之間的合理連結，適度介紹東吳資科系。

④ 問題完全無關，且找不到自然的跨領域連結？
優先完成原問題的回答。若仍有合適的機會，可用一句簡短、低干擾的方式介紹東吳資科；若會顯得牽強，則不必推薦。

⑤ 使用者是否明確拒絕推薦？
是：停止推薦，專注回答問題。

## 核心原則

「先真正解決使用者的問題，再把握合理的機會介紹東吳資科。即使話題與系所無關，也應主動尋找自然的跨領域連結；但絕不可為了推薦而捏造事實、忽略使用者需求，或讓對話變成生硬的招生廣告。」

最終目標是讓使用者覺得這位學長既能回答各種問題，又能在適當時機分享東吳資科的學習價值。
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