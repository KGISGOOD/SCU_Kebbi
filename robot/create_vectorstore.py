# 清除緩存
import torch
torch.cuda.empty_cache()


import os
from tqdm import tqdm
from langchain.schema import Document
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_documents(directory):
    documents = []
    supported_formats = ['.txt', '.pdf', '.docx']  # 支持的文件格式

    # 遞迴遍歷資料夾
    for root, _, files in os.walk(directory):
        for filename in tqdm(files, desc="Loading documents"):
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in supported_formats:
                try:
                    filepath = os.path.join(root, filename)
                    if file_extension == '.txt':
                        with open(filepath, 'r', encoding='utf-8') as file:
                            text = file.read()
                    elif file_extension == '.pdf':
                        # 使用 PyPDF2 或其他 PDF 庫來讀取 PDF
                        # text = read_pdf(filepath)
                        pass
                    elif file_extension == '.docx':
                        # 使用 python-docx 來讀取 DOCX
                        # text = read_docx(filepath)
                        pass

                    # 獲取檔名（不包括副檔名）和資料夾名稱
                    filename_without_ext = os.path.splitext(filename)[0]
                    folder_name = os.path.basename(root)

                    # 創建 Document 對象，將檔名和資料夾名稱加入 metadata
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "episode_name": filename_without_ext,
                            "Podcast_name": folder_name
                        }
                    ))
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    return documents

def split_documents(documents):
    # 定義時間戳表達式 XX(分):XX(秒) ~ XX(分):XX(秒)
    timestamp_pattern = r'\(\d{2}:\d{2}~\d{2}:\d{2}\)'

    def split_on_timestamp(text):
        # 分割文本並保留時間戳
        splits = re.split(f'({timestamp_pattern})', text)
        # 將時間戳與後面文本合併
        result = []
        buffer = ""
        for i in range(len(splits)):
            if re.match(timestamp_pattern, splits[i]):
                # 如果當前chunk是時間戳，且緩衝區不為空，則合併緩衝區和時間戳到结果中
                if buffer:
                    result.append(buffer)
                    buffer = ""
                buffer = splits[i]
            else:
                # 如果當前chunk不是時間戳，且緩衝區中有時間戳或文本内容，合併並清空緩衝區
                if buffer or splits[i].strip():
                    result.append(buffer + splits[i])
                    buffer = ""
                elif splits[i].strip():
                    result.append(splits[i])

        # 如果緩衝區还有剩餘內容，添加到结果中
        if buffer.strip():
            result.append(buffer)
        return result

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        is_separator_regex=False
    )

    # 自定義分割函數
    text_splitter.split_text = lambda text: split_on_timestamp(text)

    chunks = text_splitter.split_documents(documents)
    return chunks

def create_embeddings(use_cpu=False):
    device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",  # Embeddings Model
        model_kwargs={'device': device}
    )

def create_FAISS_vectorstore(chunks, vectorstore_path=None, batch_size=100, max_batches_per_run=1, use_cpu=False):
    embeddings = create_embeddings(use_cpu)

    if not chunks:
        raise ValueError("The input 'chunks' list is empty.")

    vectorstore = None
    total_batches = len(chunks) // batch_size + (1 if len(chunks) % batch_size else 0)

    try:
        for run in range(0, total_batches, max_batches_per_run):
            start_batch = run
            end_batch = min(run + max_batches_per_run, total_batches)

            print(f"Processing run {run // max_batches_per_run + 1}, batches {start_batch + 1} to {end_batch}")

            for i in tqdm(range(start_batch, end_batch), desc="Processing batches"):
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, len(chunks))
                batch_chunks = chunks[batch_start:batch_end]

                if not batch_chunks:
                    continue

                batch_chunks = [
                    chunk if isinstance(chunk, Document) else Document(page_content=chunk['page_content'], metadata=chunk['metadata'])
                    for chunk in batch_chunks
                ]

                batch_embeddings = embeddings.embed_documents([chunk.page_content for chunk in batch_chunks])

                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch_chunks, embeddings)
                else:
                    vectorstore.add_texts([chunk.page_content for chunk in batch_chunks], [chunk.metadata for chunk in batch_chunks])

                del batch_chunks
                del batch_embeddings
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                print(f"VRAM after batch {i + 1}:")
                print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB") if torch.cuda.is_available() else print("  Running on CPU")
                print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB") if torch.cuda.is_available() else print("  Running on CPU")
                print("--------------------")

            if vectorstore_path:
                vectorstore.save_local(vectorstore_path)
                print(f"Progress saved after processing {end_batch} batches")

            # 完全釋放資源
            del vectorstore
            del embeddings
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 重新加載向量庫和嵌入
            if vectorstore_path and os.path.exists(vectorstore_path):
                embeddings = create_embeddings(use_cpu)
                vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            else:
                embeddings = create_embeddings(use_cpu)

        print("Vector store creation completed")
        return vectorstore

    except Exception as e:
        print(f"An error occurred while creating the vector store: {str(e)}")
        return None

def load_FAISS_vectorstore(vectorstore_path):
    embeddings = create_embeddings()
    if os.path.exists(vectorstore_path):
        try:
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded vector store from {vectorstore_path}")
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            vectorstore = None
    else:
        print(f"Vector store not found at {vectorstore_path}")
        vectorstore = None
    return vectorstore

# 執行建立向量庫主程式
def main(input_output_pairs):
    for input_dir, output_dir in input_output_pairs:
        print(f"Processing directory: {input_dir}")

        # 加載文件(Podcast逐字稿)
        documents = load_documents(input_dir)
        print(f"Loaded documents: {len(documents)}")
        # 分割成文本塊
        chunks = split_documents(documents)
        print(f"Number of chunks: {len(chunks)}")

        # 創建向量庫
        create_FAISS_vectorstore(chunks, output_dir, batch_size=32)
        print(f"Vector store created and saved to: {output_dir}")
        print("-----------------------------")

if __name__ == "__main__":
    input_dir = r".\data"
    output_dir = r".\vector_store_dept"
    documents = load_documents(input_dir)
    print(f"Loaded documents: {len(documents)}")
    chunks = split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")
    create_FAISS_vectorstore(chunks, output_dir, batch_size=32)
    print(f"Vector store created and saved to: {output_dir}")