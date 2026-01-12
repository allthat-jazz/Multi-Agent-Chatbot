from dataclasses import dataclass
from pathlib import Path
import csv

from pypdf import PdfReader
from docx import Document as DocxDocument

@dataclass
class Doc:
    doc_id: str
    source: str
    text: str

def load_md_txt(path: Path):
    return path.read_text(encoding="utf-8", errors="ignore")

def load_pdf(path: Path):
    r = PdfReader(str(path))
    parts = []
    for p in r.pages:
        parts.append(p.extract_text() or "")
    return "\n".join(parts)

def load_docx(path: Path):
    d = DocxDocument(str(path))
    return "\n".join(p.text for p in d.paragraphs)

def load_csv(path: Path, max_rows=2000):
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            rows.append(" | ".join(row))
            if i >= max_rows:
                break
    return "\n".join(rows)

def load_docs_from_dir(kb_dir: Path):
    exts = {".md", ".txt", ".pdf", ".docx", ".csv"}
    docs = []
    for p in sorted(kb_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(kb_dir).as_posix()
        try:
            suf = p.suffix.lower()
            if suf in {".md", ".txt"}:
                text = load_md_txt(p)
            elif suf == ".pdf":
                text = load_pdf(p)
            elif suf == ".docx":
                text = load_docx(p)
            elif suf == ".csv":
                text = load_csv(p)
            else:
                continue
            docs.append(Doc(doc_id=rel, source=rel, text=text))
        except Exception:
            continue
    return docs
