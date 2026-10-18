"""Data loader for type1 and type2 resources.

Provides functions to parse TXT and CSV files from the
SCU_Kebbi/畢業專題_爬蟲/type1/ and type2/ directories,
clean the content, extract metadata, and convert to
LangChain Document objects.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from langchain_core.documents import Document


def _get_file_hash(filepath: str) -> str:
    """Compute MD5 hash of file content."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _clean_text(text: str) -> str:
    """Clean text: remove extra whitespace, normalize newlines, remove HTML tags."""
    if not isinstance(text, str):
        return ""
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def _parse_txt_file(filepath: str, source_group: str) -> List[Document]:
    """Parse a single TXT file."""
    try:
        # Try different encodings
        for encoding in ("utf-8-sig", "utf-8", "cp950", "big5"):
            try:
                with open(filepath, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode {filepath} with tried encodings")
    except Exception as e:
        print(f"Warning: Failed to read {filepath}: {e}")
        return []

    content = _clean_text(content)
    if not content:
        return []

    # Extract basic metadata from filename and path
    filename = os.path.basename(filepath)
    rel_path = os.path.relpath(filepath, start=os.path.join(os.path.dirname(__file__), "..", ".."))

    # Try to extract date from filename (common pattern: YYYYMM or YYYY-MM-DD)
    date_match = re.search(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})?", filename)
    published_date = None
    if date_match:
        year, month, day = date_match.groups()
        if day:
            published_date = f"{year}/{month}/{day}"
        elif month:
            published_date = f"{year}/{month}"
        else:
            published_date = year

    metadata = {
        "source_group": source_group,
        "source_file": filename,
        "source_path": rel_path,
        "content_type": _infer_content_type(filename, content),
        "title": filename.rsplit(".", 1)[0],  # remove extension
        "published_date": published_date,
        "updated_date": None,  # not available
        "category": None,
        "original_row": None,
        "document_id": f"{source_group}_{filename}_{_get_file_hash(filepath)[:8]}",
    }

    # Create a single document for the entire file
    doc = Document(page_content=content, metadata=metadata)
    return [doc]


def _parse_csv_file(filepath: str, source_group: str) -> List[Document]:
    """Parse a single CSV file."""
    try:
        # Try different encodings
        for encoding in ("utf-8-sig", "utf-8", "cp950", "big5", "gbk"):
            try:
                with open(filepath, "r", encoding=encoding, newline="") as f:
                    # Try to detect delimiter
                    sample = f.read(1024)
                    f.seek(0)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                    f.seek(0)
                    reader = csv.DictReader(f, delimiter=delimiter)
                    rows = list(reader)
                break
            except (UnicodeDecodeError, csv.Error):
                continue
        else:
            raise ValueError(f"Could not decode or parse {filepath} with tried encodings")
    except Exception as e:
        print(f"Warning: Failed to read {filepath}: {e}")
        return []

    documents = []
    filename = os.path.basename(filepath)
    rel_path = os.path.relpath(filepath, start=os.path.join(os.path.dirname(__file__), "..", ".."))

    # Infer content type from filename
    content_type = _infer_content_type(filename, "")

    for i, row in enumerate(rows, start=1):  # row numbers start at 1
        # Handle specific file structures for better content organization
        content = ""

        # Special handling for type1/ig_img.csv - single text column
        if filename == "ig_img.csv" and "text" in row:
            content = str(row["text"]).strip() if row["text"] is not None else ""

        # Special handling for type2/資料擴充 - 家長 FAQ 資料集（30 題）.csv
        elif filename == "資料擴充 - 家長 FAQ 資料集（30 題）.csv":
            question = str(row["問題內容"]).strip() if row["問題內容"] is not None else ""
            answer = str(row["ai答覆"]).strip() if row["ai答覆"] is not None else ""
            if question and answer:
                content = f"問題：{question}\n答案：{answer}"
            elif question:
                content = f"問題：{question}"
            elif answer:
                content = f"答案：{answer}"

        # Special handling for type2/資料擴充 - 相關資料搜集.csv
        elif filename == "資料擴充 - 相關資料搜集.csv":
            dimension = str(row["資料搜集維度"]).strip() if row["資料搜集維度"] is not None else ""
            content_main = str(row["完整敘述內容"]).strip() if row["完整敘述內容"] is not None else ""
            indicators = str(row["核心官方機制與客觀指標"]).strip() if row["核心官方機制與客觀指標"] is not None else ""

            # Combine all meaningful content
            parts = []
            if dimension:
                parts.append(f"維度：{dimension}")
            if content_main:
                parts.append(f"內容：{content_main}")
            if indicators:
                parts.append(f"指標：{indicators}")
            content = "\n".join(parts)

        # Default handling: combine all fields
        else:
            text_parts = []
            for key, value in row.items():
                if value is not None and str(value).strip():
                    text_parts.append(f"{key}: {str(value).strip()}")
            content = " | ".join(text_parts)

        content = _clean_text(content)
        if not content:
            continue

        # Extract metadata from row (if available)
        metadata = {
            "source_group": source_group,
            "source_file": filename,
            "source_path": rel_path,
            "content_type": content_type,
            "title": f"{filename}_row_{i}",
            "published_date": None,
            "updated_date": None,
            "category": None,
            "original_row": i,
            "document_id": f"{source_group}_{filename}_row_{i}_{_get_file_hash(filepath)[:8]}",
        }

        # Try to extract date from specific columns if they exist
        date_columns = ["日期", "时间", "time", "date", "Date", "發布日期", "更新日期"]
        for col in date_columns:
            if col in row and row[col]:
                date_val = str(row[col]).strip()
                # Try to parse common date formats
                date_match = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", date_val)
                if date_match:
                    y, m, d = date_match.groups()
                    metadata["published_date"] = f"{y}/{int(m):02d}/{int(d):02d}"
                    break
                # Try YYYYMM format
                date_match = re.search(r"(\d{4})(\d{2})", date_val)
                if date_match:
                    y, m = date_match.groups()
                    metadata["published_date"] = f"{y}/{int(m):02d}"
                    break

        # For FAQ files, preserve all original columns as metadata
        if filename == "資料擴充 - 家長 FAQ 資料集（30 題）.csv":
            metadata["編號"] = str(row["編號"]).strip() if row["編號"] is not None else ""
            metadata["關切維度"] = str(row["關切維度"]).strip() if row["關切維度"] is not None else ""
            metadata["問題內容"] = str(row["問題內容"]).strip() if row["問題內容"] is not None else ""
            metadata["ai答覆"] = str(row["ai答覆"]).strip() if row["ai答覆"] is not None else ""

        # For the data collection file, preserve metadata columns
        elif filename == "資料擴充 - 相關資料搜集.csv":
            metadata["資料搜集維度"] = str(row["資料搜集維度"]).strip() if row["資料搜集維度"] is not None else ""
            metadata["完整敘述內容"] = str(row["完整敘述內容"]).strip() if row["完整敘述內容"] is not None else ""
            metadata["核心官方機制與客觀指標"] = str(row["核心官方機制與客觀指標"]).strip() if row["核心官方機制與客觀指標"] is not None else ""

        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    return documents


def _infer_content_type(filename: str, content: str) -> str:
    """Infer content type from filename and content."""
    filename_lower = filename.lower()
    if any(keyword in filename_lower for keyword in ["faq", "常見問題", "問題"]):
        return "FAQ"
    elif any(keyword in filename_lower for keyword in ["ig", "instagram", "threads", "dcard", "社群", "貼文"]):
        return "社群貼文"
    elif any(keyword in filename_lower for keyword in ["訪談", "interview", "口稿"]):
        return "訪談"
    elif any(keyword in filename_lower for keyword in ["公告", "announcement", "notice"]):
        return "公告"
    elif any(keyword in filename_lower for keyword in ["課程", "course", "課程資訊"]):
        return "課程資訊"
    elif any(keyword in filename_lower for keyword in ["畢業", "graduate", "職涯"]):
        return "畢業資訊"
    elif any(keyword in filename_lower for keyword in ["制度", "regulation", "policy"]):
        return "制度說明"
    else:
        # Default based on content length or other heuristics
        if len(content) > 500:
            return "文章"
        else:
            return "備註"


def load_documents_from_type1_type2(
    type1_dir: str,
    type2_dir: str,
) -> Tuple[List[Document], dict]:
    """Load and parse all supported files from type1 and type2 directories.

    Args:
        type1_dir: Path to the type1 directory.
        type2_dir: Path to the type2 directory.

    Returns:
        A tuple of (documents, stats) where:
        - documents: List of LangChain Document objects.
        - stats: Dictionary with parsing statistics.
    """
    documents: List[Document] = []
    stats = {
        "type1_files": 0,
        "type2_files": 0,
        "total_documents": 0,
        "successful_parses": 0,
        "failed_parses": 0,
        "duplicates_removed": 0,
        "type1_success": 0,
        "type1_failed": 0,
        "type2_success": 0,
        "type2_failed": 0,
    }

    # Process type1 directory
    if os.path.isdir(type1_dir):
        for root, _, files in os.walk(type1_dir):
            for file in files:
                if file.lower().endswith((".txt", ".csv")):
                    filepath = os.path.join(root, file)
                    stats["type1_files"] += 1
                    try:
                        if file.lower().endswith(".txt"):
                            docs = _parse_txt_file(filepath, "type1")
                        else:
                            docs = _parse_csv_file(filepath, "type1")
                        if docs:
                            documents.extend(docs)
                            stats["successful_parses"] += 1
                            stats["type1_success"] += 1
                        else:
                            stats["failed_parses"] += 1
                            stats["type1_failed"] += 1
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
                        stats["failed_parses"] += 1
                        stats["type1_failed"] += 1
    else:
        print(f"Warning: type1 directory not found: {type1_dir}")

    # Process type2 directory
    if os.path.isdir(type2_dir):
        for root, _, files in os.walk(type2_dir):
            for file in files:
                if file.lower().endswith((".txt", ".csv")):
                    filepath = os.path.join(root, file)
                    stats["type2_files"] += 1
                    try:
                        if file.lower().endswith(".txt"):
                            docs = _parse_txt_file(filepath, "type2")
                        else:
                            docs = _parse_csv_file(filepath, "type2")
                        if docs:
                            documents.extend(docs)
                            stats["successful_parses"] += 1
                            stats["type2_success"] += 1
                        else:
                            stats["failed_parses"] += 1
                            stats["type2_failed"] += 1
                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")
                        stats["failed_parses"] += 1
                        stats["type2_failed"] += 1
    else:
        print(f"Warning: type2 directory not found: {type2_dir}")

    # Remove duplicates based on page_content
    seen_contents = set()
    unique_docs = []
    duplicates_removed = 0
    for doc in documents:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_docs.append(doc)
        else:
            duplicates_removed += 1
    stats["duplicates_removed"] = duplicates_removed
    documents = unique_docs

    stats["total_documents"] = len(documents)
    return documents, stats


if __name__ == "__main__":
    # Simple test when run directly
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    docs, stats = load_documents_from_type1_type2(
        r"..\..\畢業專題_爬蟲\type1",
        r"..\..\畢業專題_爬蟲\type2",
    )
    print(f"Loaded {len(docs)} documents")
    print(f"Stats: {stats}")