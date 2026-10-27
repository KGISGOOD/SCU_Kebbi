"""Build the FAISS vector index for the RAG baseline.

This script loads data from type1 and type2 directories, creates embeddings,
builds a FAISS index, and saves it with version information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List
from langchain_core.documents import Document

from .config import settings
from experiments.shared.data_loader import load_documents_from_type1_type2
from experiments.shared.evaluation import timer


def source_hashes():
    result = {}
    for directory in (settings.type1_dir, settings.type2_dir):
        for file in sorted(Path(directory).rglob("*")):
            if file.is_file():
                result[str(file)] = file_sha(file)
    return result


def file_sha(file):
    value = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_loaded(store, embedder, chunks, raw_documents, model_identity):
    """Reuse A5 structural contract and A4's actual chunk comparison; no LLM."""
    import numpy as np
    from experiments.a5_rag_finetuned_llm.pipeline import validate_loaded_store
    from experiments.baselines.hybrid_rag_reranker.build_bm25_index import chunk_documents as a4_chunks
    version = store._load_version_info()
    if version is None:
        raise ValueError("Dense index version.json missing/invalid; do not fabricate a version receipt")
    validate_loaded_store(store, embedder.embedding_dim, version)
    expected = sorted((d.metadata.get("document_id"), d.page_content) for d in chunks)
    actual = sorted((d.metadata.get("document_id"), d.page_content) for d in store.docstore)
    hybrid = sorted((d.metadata.get("document_id"), d.page_content)
                    for d in a4_chunks(raw_documents, chunk_size=512, chunk_overlap=128))
    if expected != actual or actual != hybrid:
        raise ValueError("Dense docstore differs from current A0/A4 source chunks; do not mix indexes")
    for start in range(0, store.index.ntotal, 1024):
        vectors = store.index.reconstruct_n(start, min(1024, store.index.ntotal - start))
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-4):
            raise ValueError("Dense vectors are nonfinite or not L2 normalized")
    receipt = store.index_path / "dense_build_receipt.json"
    if receipt.is_file():
        value = json.loads(receipt.read_text(encoding="utf-8-sig"))
        if value["model_files_sha256"] != model_identity["files_sha256"]:
            raise ValueError("Index BGE-M3 inference files differ from requested local model")
        if value["source_sha256"] != source_hashes():
            raise ValueError("Knowledge source bytes changed since index build")
        if any(file_sha(store.index_path / name) != digest for name, digest in value["index_files_sha256"].items()):
            raise ValueError("Index files changed since build receipt")
    return {"status": "passed", "vectors": store.index.ntotal, "dimension": store.index.d,
            "index_type": type(store.index).__name__, "a0_a4_chunks_equal": True,
            "a5_structural_contract": "passed", "l2_norms": "passed",
            "model_receipt_verified": receipt.is_file(), "llm_loaded": False}


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 128,
) -> List[Document]:
    """Split documents into chunks while preserving metadata.

    Args:
        documents: List of Document objects.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.

    Returns:
        List of chunked Document objects.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )

    chunked_docs = []
    for doc in documents:
        # Split the document
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            # Create metadata for the chunk
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["parent_document_id"] = doc.metadata.get(
                "document_id", f"doc_{len(chunked_docs)}"
            )
            # Update document_id to be unique for this chunk
            chunk_metadata["document_id"] = (
                f"{chunk_metadata.get('document_id', 'doc')}_chunk_{i}"
            )

            # Create chunk document
            chunked_doc = Document(page_content=chunk, metadata=chunk_metadata)
            chunked_docs.append(chunked_doc)

    return chunked_docs


def main():
    """Main function to build the vector index."""
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index for RAG baseline"
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild even if valid index exists",
    )
    parser.add_argument("--embedding-model-path", type=Path,
                        help="existing original BGE-M3 local directory; version retains BAAI/bge-m3 protocol label")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=settings.llm_device)
    parser.add_argument("--offline", action="store_true", help="never download; local checkpoint/cache only")
    parser.add_argument("--index-dir", type=Path, default=Path(settings.vector_store_dir))
    parser.add_argument("--check-only", action="store_true", help="load and validate existing index only; never build/overwrite")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"Chunk size (default: {settings.chunk_size})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"Chunk overlap (default: {settings.chunk_overlap})",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    directory = args.index_dir.expanduser()
    directory = (directory if directory.is_absolute() else root / directory).resolve()
    if (root / "experiments/vector_stores").resolve() not in directory.parents:
        parser.error("index-dir must be a child of this project's experiments/vector_stores")
    if args.check_only and args.force_rebuild:
        parser.error("check-only cannot be combined with force-rebuild")
    if not all(Path(p).is_dir() for p in (settings.type1_dir, settings.type2_dir)):
        parser.error("both original type1 and type2 directories must exist")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Delay all HF/ST imports until the offline environment is installed.
    from experiments.shared.embeddings import BGEM3Embedder
    from experiments.shared.vector_store import FAISSVectorStore
    from experiments.evaluation.model_resources import local_snapshot, inference_identity
    from experiments.a2_faq_finetuning.common import read_checkpoint_provenance
    model_path = args.embedding_model_path
    if model_path is not None:
        model_path = (model_path if model_path.is_absolute() else root / model_path).resolve()
    else:
        model_path = local_snapshot(root, settings.embedding_model)
    if not model_path.is_dir():
        parser.error("original LOCAL BGE-M3 directory missing; no download")
    if read_checkpoint_provenance(model_path) is not None:
        parser.error("Dense A0/A4/A5 knowledge index requires original BGE-M3, not an A2-trained model")
    model_identity = inference_identity(model_path, sentence_model=True)
    before_sources = source_hashes()
    index_names = ("index.faiss", "index.pkl", "version.json", "dense_build_receipt.json")
    existing = [name for name in index_names if (directory / name).exists()]
    if args.check_only and not all((directory / name).is_file() for name in index_names[:3]):
        parser.error("check-only needs existing index.faiss, index.pkl and version.json")

    print("=" * 60)
    print("Validating existing FAISS index (no build)" if args.check_only else "Building FAISS Vector Index for RAG Baseline")
    print("=" * 60)

    # Load raw documents
    print("\n1. Loading documents from type1 and type2 directories...")
    with timer("Document loading"):
        raw_documents, stats = load_documents_from_type1_type2(
            settings.type1_dir, settings.type2_dir
        )
    print(f"   Loaded {stats['total_documents']} raw documents")
    print(f"   Type1 files: {stats['type1_files']} (success: {stats['type1_success']}, failed: {stats['type1_failed']})")
    print(f"   Type2 files: {stats['type2_files']} (success: {stats['type2_success']}, failed: {stats['type2_failed']})")
    print(f"   Duplicates removed: {stats['duplicates_removed']}")

    if stats["failed_parses"]:
        raise ValueError("source parse failures; refusing a partial Dense index: %r" % stats)
    if stats["total_documents"] == 0:
        print("ERROR: No documents loaded. Please check the data directories.")
        sys.exit(1)

    # Chunk documents
    print(f"\n2. Chunking documents (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})...")
    with timer("Document chunking"):
        chunked_documents = chunk_documents(
            raw_documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    print(f"   Created {len(chunked_documents)} chunks from {len(raw_documents)} documents")

    # Initialize embedder
    print("\n3. Initializing BGE-M3 embedder...")
    with timer("Embedder initialization"):
        embedder = BGEM3Embedder(
            model_name=str(model_path),
            device=args.device,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=settings.embedding_normalize,
        )
    print(f"   Model: {settings.embedding_model}")
    print(f"   Device: {embedder.device}")
    print(f"   Embedding dimension: {embedder.embedding_dim}")

    # Initialize vector store
    print("\n4. Initializing vector store...")
    vector_store = FAISSVectorStore(
        embedding_dim=embedder.embedding_dim,
        index_path=str(directory),
        use_hnsw=settings.use_hnsw,
        hnsw_M=settings.hnsw_M,
        hnsw_efConstruction=settings.hnsw_efConstruction,
    )

    # Check if we can load existing index
    if not args.force_rebuild and vector_store.load():
        # Validate the existing index
        if vector_store.is_valid(
            documents=chunked_documents,
            embedding_model=settings.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            source_directories=[settings.type1_dir, settings.type2_dir],
        ):
            print("   Valid existing index found and loaded.")
            print(f"   Index contains {vector_store.get_document_count()} documents.")
            print(json.dumps(verify_loaded(vector_store, embedder, chunked_documents, raw_documents, model_identity), ensure_ascii=False))
            if source_hashes() != before_sources or inference_identity(model_path, sentence_model=True) != model_identity:
                raise ValueError("sources/model changed during validation")
            return
        else:
            raise ValueError("Existing index failed validation; no automatic overwrite. Use a new --index-dir or explicitly review --force-rebuild")
    if args.check_only:
        raise ValueError("Existing Dense index could not be loaded/validated; check-only never builds")
    if existing and not args.force_rebuild:
        raise ValueError("Partial/index files already exist; no automatic overwrite: %r" % existing)

    # Generate embeddings
    print(f"\n5. Generating embeddings for {len(chunked_documents)} chunks...")
    texts = [doc.page_content for doc in chunked_documents]
    with timer("Embedding generation"):
        embeddings = embedder.encode(texts, batch_size=settings.embedding_batch_size)
    print(f"   Generated embeddings shape: {embeddings.shape}")

    # Build index
    print("\n6. Building FAISS index...")
    with timer("Index building"):
        vector_store.build(embeddings, chunked_documents)
    print(f"   Index type: {'IndexHNSWFlat' if settings.use_hnsw else 'IndexFlatIP'}")
    print(f"   Index contains {vector_store.get_document_count()} vectors")

    # Save index and version info
    if source_hashes() != before_sources or inference_identity(model_path, sentence_model=True) != model_identity:
        raise ValueError("sources/model changed while generating embeddings; index not saved")
    print("\n7. Saving index and version information...")
    with timer("Index saving"):
        vector_store.save()
        vector_store._save_version_info(
            documents=chunked_documents,
            embedding_model=settings.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            source_directories=[settings.type1_dir, settings.type2_dir],
        )
    # New builds must round-trip through the same loader and consumer contracts.
    reloaded = FAISSVectorStore(embedder.embedding_dim, str(directory))
    if not reloaded.load():
        raise ValueError("saved Dense index failed to reload")
    # A force rebuild must not validate against an older receipt before it is replaced.
    validation = verify_loaded(reloaded, embedder, chunked_documents, raw_documents, model_identity) if not (directory / "dense_build_receipt.json").exists() else None
    receipt = {"schema_version": "a0_dense_build_v1", "embedding_protocol": settings.embedding_model,
               "model_path": str(model_path), "model_files_sha256": model_identity["files_sha256"],
               "source_sha256": before_sources, "loader_stats": stats,
               "index_files_sha256": {n: file_sha(directory / n) for n in index_names[:3]},
               "chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap,
               "builder_sha256": file_sha(Path(__file__)), "trained_model_modified": False}
    with (directory / "dense_build_receipt.json").open("w" if args.force_rebuild else "x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    validation = verify_loaded(reloaded, embedder, chunked_documents, raw_documents, model_identity)
    print(json.dumps(validation, ensure_ascii=False))
    print(f"   Index saved to: {directory}")
    print("   Version information saved.")

    # Final stats
    version_info = vector_store._load_version_info()
    if version_info:
        print("\n8. Index build completed successfully!")
        print(f"   Embedding model: {version_info.get('embedding_model')}")
        print(f"   Embedding dimension: {version_info.get('embedding_dimension')}")
        print(f"   Chunk size: {version_info.get('chunk_size')}")
        print(f"   Chunk overlap: {version_info.get('chunk_overlap')}")
        print(f"   Source directories: {version_info.get('source_directories')}")
        print(f"   Document count: {version_info.get('document_count')}")
        print(f"   Chunk count: {version_info.get('chunk_count')}")
        print(f"   Index type: {version_info.get('index_type')}")
        print(f"   Data fingerprint: {version_info.get('data_fingerprint')[:16]}...")

    print("\n" + "=" * 60)
    print("Index building completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
