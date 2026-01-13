import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from .loaders import load_docs_from_dir
from .store import (save_chunks, load_chunks,
    save_bm25_tokens, load_bm25_tokens,
    save_faiss, load_faiss)

@dataclass
class Chunk:
    doc_id: str
    source: str
    chunk_id: str
    title: str
    text: str

_heading_re = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

def _split_md_by_headings(text):
    matches = list(_heading_re.finditer(text))
    if not matches:
        return [("Document", text.strip())]
    out = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = (m.group(2) or "").strip() or "Section"
        body = text[start:end].strip()
        out.append((title, body))
    return out

def _chunk_text(title, body, max_chars, overlap):
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return []
    chunks = []
    i = 0
    n = len(body)
    while i < n:
        j = min(i + max_chars, n)
        piece = body[i:j].strip()
        if piece:
            chunks.append((title, piece))
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks

def _tokenize(text):
    return re.findall(r"[a-zA-Zа-яА-Я0-9_]+", text.lower())

def _normalize_scores(scores):
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if abs(mx - mn) < 1e-9:
        return [0.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]

class HybridRAG:
    def __init__(
        self,
        kb_dir: Path,
        index_dir: Path,
        emb_model: str,
        chunk_max_chars: int,
        chunk_overlap_chars: int,
        hybrid_alpha: float,
        candidates: int,
        use_rerank: bool,
        rerank_model: str,
        rerank_topn: int,
    ):
        self.kb_dir = kb_dir
        self.index_dir = index_dir
        self.emb_model = emb_model
        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self.hybrid_alpha = hybrid_alpha
        self.candidates = candidates
        self.use_rerank = use_rerank
        self.rerank_model = rerank_model
        self.rerank_topn = rerank_topn

        self._reranker = None
        self._embedder = None
        self._chunks = []
        self._faiss = None
        self._bm25 = None
        self._bm25_tokens = None

    def _ensure_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.emb_model)
        return self._embedder
    
    def _ensure_reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(self.rerank_model)
        return self._reranker

    def load_if_exists(self):
        meta = load_chunks(self.index_dir)
        toks = load_bm25_tokens(self.index_dir)
        fx = load_faiss(self.index_dir)

        if not meta or toks is None:
            return False

        self._chunks = [Chunk(**m) for m in meta]
        self._bm25_tokens = toks
        self._bm25 = BM25Okapi(self._bm25_tokens)
        self._faiss = fx
        return True

    def reindex(self):
        docs = load_docs_from_dir(self.kb_dir)

        chunks = []
        for d in docs:
            local_i = 0
            sections = _split_md_by_headings(d.text)
            for title, sec in sections:
                for title2, piece in _chunk_text(title, sec, self.chunk_max_chars, self.chunk_overlap_chars):
                    cid = f"{d.doc_id}::c{local_i:04d}"
                    local_i += 1
                    chunks.append(Chunk(
                        doc_id=d.doc_id,
                        source=d.source,
                        chunk_id=cid,
                        title=title2,
                        text=piece,
                    ))

        self._chunks = chunks
        texts = [c.text for c in chunks]

        if not texts:
            save_chunks(self.index_dir, [])
            save_bm25_tokens(self.index_dir, [])
            self._faiss = None
            self._bm25 = BM25Okapi([[]])
            self._bm25_tokens = [[]]
            return {"docs": 0, "chunks": 0}

        emb = self._ensure_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        self._faiss = index

        bm25_tokens = [_tokenize(t) for t in texts]
        self._bm25_tokens = bm25_tokens
        self._bm25 = BM25Okapi(bm25_tokens)

        save_chunks(self.index_dir, [c.__dict__ for c in chunks])
        save_bm25_tokens(self.index_dir, bm25_tokens)
        save_faiss(self.index_dir, index)

        return {"docs": len({c.doc_id for c in chunks}), "chunks": len(chunks)}

    def search(self, query, k=5):
        if not self._chunks:
            self.load_if_exists()

        if not self._chunks or (self._faiss is None and self._bm25 is None):
            return []

        sem_ids = []
        sem_scores = []
        if self._faiss is not None:
            qemb = self._ensure_embedder().encode([query], normalize_embeddings=True, show_progress_bar=False)
            qemb = np.asarray(qemb, dtype=np.float32)
            D, I = self._faiss.search(qemb, min(self.candidates, len(self._chunks)))
            sem_ids = I[0].tolist()
            sem_scores = D[0].tolist()

        lex_ids = []
        lex_scores = []
        if self._bm25 is not None and self._bm25_tokens is not None:
            qtoks = _tokenize(query)
            scores = self._bm25.get_scores(qtoks)
            top = np.argsort(scores)[::-1][: min(self.candidates, len(self._chunks))]
            lex_ids = top.tolist()
            lex_scores = [float(scores[i]) for i in lex_ids]

        cand_set = set(sem_ids) | set(lex_ids)
        if not cand_set:
            return []

        sem_map = {i: s for i, s in zip(sem_ids, sem_scores)}
        lex_map = {i: s for i, s in zip(lex_ids, lex_scores)}

        sem_pool = [sem_map.get(i, 0.0) for i in cand_set]
        lex_pool = [lex_map.get(i, 0.0) for i in cand_set]
        sem_norm = _normalize_scores(sem_pool)
        lex_norm = _normalize_scores(lex_pool)

        cand_list = list(cand_set)
        hits = []
        for idx, i in enumerate(cand_list):
            s_sem = sem_norm[idx]
            s_lex = lex_norm[idx]
            score = self.hybrid_alpha * s_sem + (1 - self.hybrid_alpha) * s_lex
            c = self._chunks[i]
            hits.append({
                "source": c.source,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "title": c.title,
                "text": c.text,
                "score": float(score),
                "score_sem": float(s_sem),
                "score_lex": float(s_lex),
            })

        hits.sort(key=lambda x: x["score"], reverse=True)

        if self.use_rerank and hits:
            topn = min(max(1, self.rerank_topn), len(hits))
            subset = hits[:topn]
            try:
                pairs = [[query, h["text"]] for h in subset]
                r_scores = self._ensure_reranker().predict(pairs)
                for i, s in enumerate(r_scores):
                    subset[i]["rerank_score"] = float(s)
                subset.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
                hits = subset + hits[topn:]
            except Exception:
                pass
        return hits[:k]

