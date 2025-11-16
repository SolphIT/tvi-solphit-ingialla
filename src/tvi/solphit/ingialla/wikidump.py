from __future__ import annotations
import os
import hashlib
from lxml import etree
from tqdm import tqdm
from tvi.solphit.ingialla.es import ensure_articles_index, already_split, mark_split_done
from tvi.solphit.base.logging import SolphitLogger

log = SolphitLogger.get_logger("tvi.solphit.ingialla.wikidump")

WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1,10)), *(f"LPT{i}" for i in range(1,10))
}

def _hash8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def safe_filename(title: str, max_len: int = 120) -> str:
    if not title or title == "":
        title = "untitled"
    cleaned = "".join(c if (c.isalnum() or c in " ._-") else "_" for c in title)
    cleaned = " ".join(cleaned.split()).strip()
    base, ext = os.path.splitext(cleaned)
    if base.upper() in WINDOWS_RESERVED:
        base = f"{base}_file"
    cleaned = (base + ext).rstrip(" .")
    suffix = f"-{_hash8(cleaned)}"
    max_len = max(8 + len(suffix), max_len)
    if len(cleaned) > max_len - len(suffix):
        cleaned = cleaned[: max_len - len(suffix)]
        cleaned = cleaned.rstrip(" .")
    return f"{cleaned}{suffix}" if not cleaned.endswith(suffix) else cleaned

def _hashed_path(root: str, key: str, depth: int = 5) -> str:
    h = hashlib.sha1(key.encode('utf-8')).hexdigest()
    parts = [h[i] for i in range(depth)]
    return os.path.join(root, *parts)

def get_article_path(base_dir: str, title: str, ext: str = ".xml", max_path: int = 240) -> str:
    folder = _hashed_path(base_dir, title, depth=5)
    os.makedirs(folder, exist_ok=True)
    name = safe_filename(title, max_len=120)
    candidate = os.path.join(folder, f"{name}{ext}")
    if len(candidate) >= max_path:
        usable_len = max(24, max_path - len(folder) - 1 - len(ext))
        name = safe_filename(title, max_len=usable_len)
        candidate = os.path.join(folder, f"{name}{ext}")
    return candidate

def extract_articles(xml_path: str, output_dir: str, max_pages: int | None = None) -> int:
    es = ensure_articles_index()
    file_size = os.path.getsize(xml_path)
    pages_saved = 0
    with open(xml_path, 'rb') as f:
        context = etree.iterparse(f, events=('end',), tag='{*}page')
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc='Processing XML')
        for _, elem in context:
            if max_pages is not None and pages_saved >= max_pages:
                break
            title_elem = elem.find('./{*}title')
            title = title_elem.text if title_elem is not None else 'untitled'
            path = get_article_path(output_dir, title)
            if already_split(es, path):
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                continue
            data = etree.tostring(elem, encoding='utf-8')
            try:
                with open(path, 'wb') as out_file:
                    out_file.write(data)
                pages_saved += 1
                log.info(f"[{pages_saved}] Saved '{title}' -> {path}")
                mark_split_done(es, path, title)
            except OSError as e:
                log.warning(f"[SKIP] Could not save '{title}' due to: {e}")
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        pbar.close()
        log.info(f"Completed. Total pages saved: {pages_saved}")
    return pages_saved