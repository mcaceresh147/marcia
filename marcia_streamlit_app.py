"""
Marcia News Curator — v2
=======================

*Streamlit GUI* + *CLI* para generar el bloque HTML de **Destacados de la Semana**.

### Ejecución local
```bash
streamlit run marcia_streamlit_app.py
```

### CLI rápido
```bash
python marcia_streamlit_app.py --url https://ejemplo.com
python marcia_streamlit_app.py --img captura.png
python marcia_streamlit_app.py --generate-html
```

Un archivo oculto `.marcia_cache.json` mantiene hasta diez noticias.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Importaciones opcionales ───────────────────────────────────────────────
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    st = None  # CLI‑only

import pytesseract
from PIL import Image
from dateutil import parser as dateparser

# Noticias via newspaper3k → fallback requests+BS4
try:
    from newspaper import Article
except ImportError:
    Article = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    BeautifulSoup = None

try:
    import requests
except ImportError:
    requests = None  # fallback no disponible

# OpenAI opcional
try:
    import openai  # type: ignore

    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    if OPENAI_AVAILABLE:
        openai.api_key = os.getenv("OPENAI_API_KEY")
except ImportError:
    OPENAI_AVAILABLE = False

# ── Constantes ─────────────────────────────────────────────────────────────
TEMPLATE = (
    "<div class=\"news-item\">\n"
    "  <a class=\"news-link\" href=\"{url}\">{title}</a>\n"
    "  <div class=\"news-details\">{source} — {date}</div>\n"
    "</div>"
)
GROUP_ORDER = {"econpol": 0, "coyuntural": 1, "opinion": 2}
CACHE_FILE = Path(".marcia_cache.json")

# ── Utilidades ─────────────────────────────────────────────────────────────

def classify_article(title: str, body: str) -> str:
    """Return 'econpol', 'coyuntural' or 'opinion'."""
    if OPENAI_AVAILABLE:
        prompt = (
            "Classify climate‑related news as ECONPOL, COYUNTURAL or OPINION.\n"
            f"Headline: {title}\nExcerpt: {body[:400]}"
        )
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
            )
            lbl = resp.choices[0].message.content.strip().upper()
            if lbl in ("ECONPOL", "COYUNTURAL", "OPINION"):
                return lbl.lower()
        except Exception:
            pass
    if re.search(r"\b(opinión|opinion|column|editorial)\b", title, re.I):
        return "opinion"
    if re.search(r"\b(finanzas|mercado|banco|gobierno|economía|impuesto|policy|política|regulación)\b", title, re.I):
        return "econpol"
    return "coyuntural"


def guess_date(s: str | None):
    try:
        return dateparser.parse(str(s), dayfirst=False).date() if s else None
    except Exception:
        return None


def _open_image(src: Any):
    if isinstance(src, (bytes, bytearray)):
        return Image.open(BytesIO(src))
    if hasattr(src, "read"):
        return Image.open(src)
    return Image.open(Path(src))

# ── Extractores ────────────────────────────────────────────────────────────

def extract_from_image(src: Any) -> Tuple[Dict[str, Any] | None, str]:
    """OCR → dict | error."""
    try:
        img = _open_image(src)
        txt = pytesseract.image_to_string(img, lang="spa+eng")
    except pytesseract.TesseractNotFoundError:
        return None, "Tesseract OCR no está instalado en el servidor."
    except Exception as e:
        return None, f"OCR error: {e}"

    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if not lines:
        return None, "No se encontró texto legible en la imagen."
    title = lines[0]
    date = next((d for l in lines if (d := guess_date(l))), None) or datetime.today().date()
    return {
        "title": title,
        "source": "Desconocido",
        "date": date,
        "url": "",
        "raw_text": txt,
    }, ""


def _scrape_fallback(url: str):
    """Scrape with requests + BeautifulSoup when newspaper3k fails."""
    if requests is None or BeautifulSoup is None:
        return None, "Dependencias de scraping no instaladas."
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title_tag = soup.find("meta", property="og:title") or soup.find("title")
        title = title_tag["content"] if title_tag and title_tag.get("content") else title_tag.get_text(strip=True)
        body = soup.get_text(" ", strip=True)[:2000]
        site = soup.find("meta", property="og:site_name")
        source = site["content"] if site and site.get("content") else url.split("/")[2]
        date_meta = soup.find("meta", property="article:published_time")
        pub_date = guess_date(date_meta["content"] if date_meta and date_meta.get("content") else None) or datetime.today().date()
        return {
            "title": title or "Untitled",
            "source": source,
            "date": pub_date,
            "url": url,
            "raw_text": body,
        }, ""
    except Exception as e:
        return None, f"Scraping fallback error: {e}"


def extract_from_url(url: str) -> Tuple[Dict[str, Any] | None, str]:
    if Article is None:
        return _scrape_fallback(url)
    try:
        art = Article(url, language="es")
        art.download(); art.parse()
    except Exception:
        return _scrape_fallback(url)

    title = art.title or "Untitled"
    text = art.text or ""
    soup = BeautifulSoup(art.html, "html.parser") if BeautifulSoup else None
    source = (
        soup.find("meta", property="og:site_name").get("content")  # type: ignore[attr-defined]
        if soup and soup.find("meta", property="og:site_name") else url.split("/")[2]
    )
    date_meta = soup.find("meta", property="article:published_time") if soup else None  # type: ignore[attr-defined]
    pub_date = guess_date(date_meta.get("content") if date_meta else None) or guess_date(art.publish_date) or datetime.today().date()
    return {
        "title": title,
        "source": source,
        "date": pub_date,
        "url": url,
        "raw_text": text,
    }, ""

# ── Persistencia ───────────────────────────────────────────────────────────

def load_cache() -> List[Dict[str, Any]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return []


def save_cache(data: List[Dict[str, Any]]):
    CACHE_FILE.write_text(json.dumps(data, default=str, ensure_ascii=False, indent=2))

# ── Core ───────────────────────────────────────────────────────────────────

def add_item(storage: List[Dict[str, Any]], item: Dict[str, Any]):
    item["group"] = classify_article(item["title"], item.get("raw_text", ""))
    storage.append(item)
    storage.sort(key=lambda n: (GROUP_ORDER[n["group"]], -n["date"].toordinal()))
    del storage[10:]


def generate_html_block(storage: List[Dict[str, Any]]) -> str:
    ordered = sorted(storage, key=lambda n: (GROUP_ORDER[n["group"]], -n["date"].toordinal()))
    return "\n".join(
        TEMPLATE.format(url=n["url"] or "#", title=n["title"], source=n["source"], date=n["date"].strftime("%d %b %Y"))
        for n in ordered
    )

# ── CLI ────────────────────────────────────────────────────────────────────

def cli_main(argv: List[str]):
    p = argparse.ArgumentParser()
    p.add_argument("--url")
    p.add_argument("--img")
    p.add_argument("--generate-html", action="store_true")
    args = p.parse_args(argv)

    items = load_cache()
    if args.url:
        it, err = extract_from_url(args.url)
        print(err or "URL added.")
        if it:
            add_item(items, it)
    if args.img:
        it, err = extract_from_image(Path(args.img))
        print(err or "Image added.")
        if it:
            add_item(items, it)
    save_cache(items)
    if args.generate_html:
        print(generate_html_block(items))

# ── GUI ────────────────────────────────────────────────────────────────────

def gui_main():
    st.set_page_config(page_title="Marcia – Climate News Curator", layout="wide")
    st.title("📰 Marcia – Climate News Curator")

    if "news_items" not in st.session_state:
        st.session_state.news_items = load_cache()
    if "log" not in st.session_state:
        st.session_state.log = []  # type: ignore[attr-defined]

    def log(msg: str, error=False):  # helper dentro de GUI
        st.session_state.log.append((msg, error))  # type: ignore[attr-defined]

    with st.sidebar:
        st.subheader("Add news item")
        upload = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "pdf"])
        url_input = st.text_input("Paste URL")

        if st.button("Add Image"):
            if upload is None:
                st.warning("No image selected.")
            else:
                item, err
