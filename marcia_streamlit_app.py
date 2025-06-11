"""
Marcia News Curator — v3.1 (cloud‑ready, fixed)
==============================================

Funciona localmente y en **Streamlit Community Cloud**.

### Usar la app gráfica
```bash
streamlit run marcia_streamlit_app.py
```

### CLI rápido
```bash
python marcia_streamlit_app.py --url https://ejemplo.com
python marcia_streamlit_app.py --img captura.png
python marcia_streamlit_app.py --generate-html
```
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

# ── Importaciones condicionales ───────────────────────────────────────────
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    st = None  # CLI‑only

# OCR ----------------------------------------------------------------------
try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
    from PIL import Image
    try:
        pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
    except TesseractNotFoundError:
        OCR_AVAILABLE = False
except ImportError:
    OCR_AVAILABLE = False

# Scraping -----------------------------------------------------------------
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
    import requests_cache
    requests_cache.install_cache("marcia_http", expire_after=3600)
except ImportError:
    requests = None  # type: ignore

from dateutil import parser as dateparser

# OpenAI -------------------------------------------------------------------
try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

api_key = (
    os.getenv("OPENAI_API_KEY")
    or (st.secrets["OPENAI_API_KEY"] if st and "OPENAI_API_KEY" in st.secrets else "")
)
OPENAI_AVAILABLE = bool(openai and api_key)
if OPENAI_AVAILABLE:
    openai.api_key = api_key

# ── Constantes ────────────────────────────────────────────────────────────
TEMPLATE = (
    "<div class=\"news-item\">\n"
    "  <a class=\"news-link\" href=\"{url}\">{title}</a>\n"
    "  <div class=\"news-details\">{source} — {date}</div>\n"
    "</div>"
)
GROUP_ORDER = {"econpol": 0, "coyuntural": 1, "opinion": 2}
CACHE_FILE = Path(".marcia_cache.json")

# ── Utilidades ------------------------------------------------------------

def classify_article(title: str, body: str) -> str:
    if OPENAI_AVAILABLE:
        prompt = (
            "Classify the following climate news as ECONPOL, COYUNTURAL or OPINION.\n"
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

# ── Extractores -----------------------------------------------------------

def _open_image(src: Any):
    if isinstance(src, (bytes, bytearray)):
        return Image.open(BytesIO(src))
    if hasattr(src, "read"):
        return Image.open(src)
    return Image.open(Path(src))


def extract_from_image(src: Any) -> Tuple[Dict[str, Any] | None, str]:
    if not OCR_AVAILABLE:
        return None, "OCR no disponible en el servidor. Usa URL."
    try:
        img = _open_image(src)
        txt = pytesseract.image_to_string(img, lang="spa+eng")
    except Exception as e:
        return None, f"OCR error: {e}"
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if not lines:
        return None, "No se encontró texto legible en la imagen."
    title = lines[0]
    date = next((d for l in lines if (d := guess_date(l))), None) or datetime.today().date()
    return {"title": title, "source": "Desconocido", "date": date, "url": "", "raw_text": txt}, ""


def _scrape_fallback(url: str):
    if not (requests and BeautifulSoup):
        return None, "Dependencias de scraping faltantes."
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title_tag = soup.find("meta", property="og:title") or soup.find("title")
        title = (
            title_tag.get("content") if title_tag and title_tag.get("content") else title_tag.get_text(strip=True)
        ) if title_tag else "Untitled"
        body = soup.get_text(" ", strip=True)[:2000]
        source_tag = soup.find("meta", property="og:site_name")
        source = source_tag.get("content") if source_tag and source_tag.get("content") else url.split("/")[2]
        date_meta = soup.find("meta", property="article:published_time")
        pub_date = guess_date(date_meta.get("content") if date_meta and date_meta.get("content") else None) or datetime.today().date()
        return {"title": title, "source": source, "date": pub_date, "url": url, "raw_text": body}, ""
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
        soup.find("meta", property="og:site_name").get("content") if soup and soup.find("meta", property="og:site_name") else url.split("/")[2]
    )
    date_meta = soup.find("meta", property="article:published_time") if soup else None
    pub_date = guess_date(date_meta.get("content") if date_meta else None) or guess_date(art.publish_date) or datetime.today().date()
    return {"title": title, "source": source, "date": pub_date, "url": url, "raw_text": text}, ""

# ── Persistencia ----------------------------------------------------------

def load_cache() -> List[Dict[str, Any]]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return []


def save_cache(data: List[Dict[str, Any]]):
    CACHE_FILE.write_text(json.dumps(data, default=str, ensure_ascii=False, indent=2))

# ── Core helpers ----------------------------------------------------------

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

# ── CLI -------------------------------------------------------------------

def cli_main(argv: List[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--url")
    parser.add_argument("--img")
    parser.add_argument("--generate-html", action="store_true")
    args = parser.parse_args(argv)

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

# ── Streamlit GUI ---------------------------------------------------------

def gui_main():
    st.set_page_config(page_title="Marcia – Climate News Curator", layout="wide")
    st.title("📰 Marcia – Climate News Curator")

    if "news_items" not in st.session_state:
        st.session_state.news_items = load_cache()
    if "log" not in st.session_state:
        st.session_state.log: List[Tuple[str, bool]] = []

    def log(msg: str, error: bool = False):
        st.session_state.log.append((msg, error))

    # Sidebar input
    with st.sidebar:
        st.header("Add news item")
        upload = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "pdf"])
        url_input = st.text_input("Paste URL")

        if st.button("Add Image"):
            if upload is None:
                log("No image selected.", True)
            else:
                item, err = extract_from_image(upload.getvalue())
                if err:
                    log(err, True)
                elif item:
                    add_item(st.session_state.news_items, item)
                    log("Image added.")

        if st.button("Add URL"):
            if not url_input.strip():
                log("URL field is empty.", True)
            else:
                item, err = extract_from_url(url_input.strip())
                if err:
                    log(err, True)
                elif item:
                    add_item(st.session_state.news_items, item)
                    log("URL added.")

        st.subheader("Logs")
        for m, is_err in st.session_state.log:
            (st.error if is_err else st.success)(m)

    # Main table
    st.subheader("Current roster")
    if st.session_state.news_items:
        tbl = [{
            "Title": n["title"],
            "Source": n["source"],
            "Date": n["date"].strftime("%Y-%m-%d"),
            "Group": n["group"],
            "URL": n["url"],
        } for n in st.session_state.news_items]
        st.dataframe(tbl, use_container_width=True)
    else:
        st.info("No items yet.")

    if st.button("Generate HTML"):
        html = generate_html_block(st.session_state.news_items)
        st.code(html, language="html")
        st.download_button("Download", data=html, file_name="destacados.html", mime="text/html")
        save_cache(st.session_state.news_items)

# ── Entrypoint ------------------------------------------------------------
if __name__ == "__main__":
    if st is None:
        cli_main(sys.argv[1:])
    else:
        gui_main()

# ── Tests -----------------------------------------------------------------
if __name__ == "__test__":
    assert classify_article("Opinion: Climate policy", "") == "opinion"
    assert classify_article("Gobierno presenta nueva regulación", "") == "econpol"
    assert classify_article("Se inaugura la COP30", "") == "coyuntural"
