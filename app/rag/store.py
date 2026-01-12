import json
from pathlib import Path
import faiss

def save_chunks(index_dir: Path, chunks):
    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8")

def load_chunks(index_dir: Path):
    p = index_dir / "chunks.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def save_bm25_tokens(index_dir: Path, tokens):
    (index_dir / "bm25_tokens.json").write_text(
        json.dumps(tokens, ensure_ascii=False),
        encoding="utf-8")

def load_bm25_tokens(index_dir: Path):
    p = index_dir / "bm25_tokens.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

def save_faiss(index_dir: Path, faiss_index):
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))

def load_faiss(index_dir: Path):
    p = index_dir / "faiss.index"
    if not p.exists():
        return None
    return faiss.read_index(str(p))
