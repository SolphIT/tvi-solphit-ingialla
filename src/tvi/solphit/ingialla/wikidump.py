
from __future__ import annotations
import os
import json
import time
import hashlib
from dataclasses import dataclass
from typing import Optional
from lxml import etree
from tqdm import tqdm

from tvi.solphit.ingialla.es import (
    ensure_articles_index,
    already_split,
    mark_split_done,
)
from tvi.solphit.base.logging import SolphitLogger

log = SolphitLogger.get_logger("tvi.solphit.ingialla.wikidump")

WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]
}

def _hash8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def safe_filename(title: str, max_len: int = 120) -> str:
    """Turn a wiki page title into a safe filename, suffixing a short hash to avoid collisions."""
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
    """Create a shard-like subfolder path based on a SHA1 of the key (title)."""
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
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

@dataclass
class SplitOptions:
    xml_path: str
    output_dir: str
    max_pages: Optional[int] = None
    log_every: int = 100                 # INFO/inline summary every N pages saved
    status_interval: float = 5.0         # refresh split-status.json every X seconds
    show_titles: bool = False            # emit INFO per title (else DEBUG only)
    inline_summary: bool = True          # summary refreshes inline via tqdm postfix (no new lines)

def _write_status(status_path: str, *, pages_saved: int, pages_skipped: int,
                  bytes_processed: int, file_size: int, started_at: float):
    """Write a small status JSON you can tail while long runs execute."""
    now = time.time()
    elapsed = max(0.001, now - started_at)
    pct = (bytes_processed / file_size) if file_size > 0 else 0.0
    status = {
        "pages_saved": pages_saved,
        "pages_skipped": pages_skipped,
        "bytes_processed": bytes_processed,
        "file_size": file_size,
        "progress_bytes_pct": round(pct * 100, 2),
        "elapsed_seconds": round(elapsed, 2),
        "rate_pages_per_sec": round(pages_saved / elapsed, 3),
        "rate_mib_per_sec": round((bytes_processed / (1024 * 1024)) / elapsed, 3),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
    except Exception as ex:
        log.debug("Failed to write status file %s: %s", status_path, ex)

def extract_articles(xml_path: str, output_dir: str, max_pages: int | None = None) -> int:
    """
    Backwards-compatible entry used by SplitDumpJob.
    """
    opts = SplitOptions(xml_path=xml_path, output_dir=output_dir, max_pages=max_pages)
    return extract_articles_verbose(opts)

def extract_articles_verbose(opts: SplitOptions) -> int:
    """
    Split a MediaWiki dump XML into per-article files and track progress in ES.

    Enhancements:
      - bytes-progress bar (uses file.tell())
      - pages-progress bar
      - per-title logs (optional)
      - periodic status JSON writes (split-status.json)
      - periodic summaries either as inline tqdm postfix (no new lines) or as log lines
    """
    # ES ready?
    es = ensure_articles_index()

    # Size + status file path
    try:
        file_size = os.path.getsize(opts.xml_path)
    except Exception:
        file_size = 0
    status_path = os.path.join(opts.output_dir, "split-status.json")

    # Counters
    pages_saved = 0
    pages_skipped = 0
    started_at = time.time()
    last_bytes = 0
    last_status_write = 0.0

    # Progress bars
    bytes_bar = tqdm(total=file_size or None, unit="B", unit_scale=True,
                     desc="Processing XML", leave=True, position=0)
    pages_total = opts.max_pages or None
    pages_bar = tqdm(total=pages_total, unit="page", desc="Pages saved", leave=True, position=1)
    # We’ll use pages_bar.set_postfix_str("...") for inline summaries (keeps lines stable)

    # Open & stream parse
    with open(opts.xml_path, "rb") as f:
        context = etree.iterparse(f, events=("end",), tag="{*}page")

        for _, elem in context:
            # Update BYTES progress from file.tell() (delta since last iteration)
            try:
                cur = f.tell()
                delta = max(0, cur - last_bytes)
                if delta:
                    bytes_bar.update(delta)
                    last_bytes = cur
            except Exception:
                pass

            if opts.max_pages is not None and pages_saved >= opts.max_pages:
                # We've hit the requested limit; stop early.
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                break

            # Extract title
            title_elem = elem.find("./{*}title")
            title = title_elem.text if title_elem is not None else "untitled"

            # Destination path
            path = get_article_path(opts.output_dir, title)

            # Skip if ES says already processed
            if already_split(es, path):
                pages_skipped += 1
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
                # Report skips every 500 to match your screenshot behavior
                if pages_skipped % 500 == 0:
                    if opts.inline_summary:
                        # Update inline postfix instead of printing a new line
                        pages_bar.set_postfix_str(
                            f"skipped={pages_skipped} last='{title[:40]}'"
                        )
                    else:
                        log.info("[skip] %s pages already split (latest title=%r)", pages_skipped, title)
                continue

            # Serialize and save
            try:
                data = etree.tostring(elem, encoding="utf-8")
                with open(path, "wb") as out_file:
                    out_file.write(data)
                pages_saved += 1
                pages_bar.update(1)

                # Per-title log (optional at INFO; always DEBUG)
                if opts.show_titles:
                    log.info("[save %6d] '%s' -> %s", pages_saved, title, path)
                else:
                    log.debug("[save %6d] '%s' -> %s", pages_saved, title, path)

                # Mark in ES
                mark_split_done(es, path, title)

                # Periodic summary (inline or log)
                if pages_saved % max(1, opts.log_every) == 0:
                    elapsed = max(0.001, time.time() - started_at)
                    rate = pages_saved / elapsed
                    summary = (
                        f"saved={pages_saved} skipped={pages_skipped} "
                        f"rate={rate:.2f} pages/s last='{title[:40]}'"
                    )
                    if opts.inline_summary:
                        # Single-line refresh: update postfix on the "Pages saved" bar
                        pages_bar.set_postfix_str(summary)
                    else:
                        log.info("[progress] %s", summary)

                # Periodic status file
                now = time.time()
                if (now - last_status_write) >= max(0.5, float(opts.status_interval)):
                    last_status_write = now
                    _write_status(status_path,
                                  pages_saved=pages_saved,
                                  pages_skipped=pages_skipped,
                                  bytes_processed=last_bytes,
                                  file_size=file_size,
                                  started_at=started_at)

            except OSError as e:
                log.warning("[SKIP] Could not save '%s' due to: %s", title, e)

            # Clear element to free memory and keep parser streaming
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    # Final status write
    _write_status(status_path,
                  pages_saved=pages_saved,
                  pages_skipped=pages_skipped,
                  bytes_processed=last_bytes,
                  file_size=file_size,
                  started_at=started_at)

    # Close bars at the end
    pages_bar.close()
    bytes_bar.close()

    log.info("Completed. Total pages saved: %s (skipped: %s)", pages_saved, pages_skipped)
