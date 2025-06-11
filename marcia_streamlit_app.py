"""
Marcia News Curator
===================

Streamlit GUI + CLI para curar *Destacados de la Semana*.

### Ejecución local
```bash
streamlit run marcia_streamlit_app.py
```

### CLI
```bash
python marcia_streamlit_app.py --url https://ejemplo.com
python marcia_streamlit_app.py --img screenshot.png
python marcia_streamlit_app.py --generate-html
```
Un archivo oculto `.marcia_cache.json` guarda hasta diez noticias.
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

# ── Dependencias opcionales ────────────────────────────────────────────────
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    st = None  # CLI‑only

import pytesseract
from PIL import Image

try:
    from newspaper import Article
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    Article = None
    BeautifulSoup = None

from dateutil import parser as dateparser

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
    """ECONPOL / COYUNTURAL / OPINION"""
    if OPENAI_AVAILABLE:
        prompt = (
            "Classify the following climate‑related news headline + excerpt as ECONPOL, COYUNTURAL or OPINION.\n"
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
            pass  # Fallback heuristics
    if re.search(r"\b(opinión|opinion|column|editorial)\b", title, flags=re.I):
        return "opinion"
    if re.search(r"\b(finanzas|mercado|banco|gobierno|economía|impuesto|policy|política|regulación)\b", title, flags=re.I):
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
    """Devuelve (item, error_msg)."""
    try:
        img = _open_image(src)
        txt = pytesseract.image_to_string(img, lang="spa+eng")
    except Exception as e:
        return None, f"OCR failed: {e}"

    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if not lines:
        return None, "Could not read any text in the image."
    title = lines[0]
    date = next((d for l in lines if (d := guess_date(l))), None) or datetime.today().date()
    item = {
        "title": title,
        "source": "Desconocido",
        "date": date,
        "url": "",
        "raw_text": txt,
    }
    return item, ""


def extract_from_url(url: str) -> Tuple[Dict[str, Any] | None, str]:
    if Article is None:
        return None, "newspaper3k is not installed."
    try:
        art = Article(url, language="es")
        art.download(); art.parse()
    except Exception as e:
        return None, f"Download error: {e}"

    title = art.title or "Untitled"
    text = art.text or ""
    soup = BeautifulSoup(art.html, "html.parser") if BeautifulSoup else None
    src = (
        soup.find("meta", property="og:site_name").get("content")  # type: ignore[attr-defined]
        if soup and soup.find("meta", property="og:site_name") else url.split("/")[2]
    )
    date_meta = soup.find("meta", property="article:published_time") if soup else None  # type: ignore[attr-defined]
    pub_date = guess_date(date_meta.get("content") if date_meta else None) or guess_date(art.publish_date) or datetime.today().date()

    item = {"title": title, "source": src, "date": pub_date, "url": url, "raw_text": text}
    return item, ""

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

    with st.sidebar:
        st.subheader("Add news item")
        upload = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "pdf"])
        url_input = st.text_input("Paste URL")

        if st.button("Add Image"):
            if upload is None:
                st.warning("No image selected.")
            else:
                item, err = extract_from_image(upload.getvalue())
                if err:
                    st.error(err)
                else:
                    add_item(st.session_state.news_items, item)
                    st.success("Image added.")

        if st.button("Add URL"):
            if not url_input.strip():
                st.warning("URL field is empty.")
            else:
                item, err = extract_from_url(url_input.strip())
                if err:
                    st.error(err)
                else:
                    add_item(st.session_state.news_items, item)
                    st.success("URL added.")

    st.subheader("Current roster")
    df = [
        {
            "Title": n["title"],
            "Source": n["source"],
            "Date": n["date"].strftime("%Y-%m-%d"),
            "Group": n["group"],
            "URL": n["url"],
        }
        for n in st.session_state.news_items
    ]
    st.dataframe(df, use_container_width=True)

    if st.button("Generate HTML"):
        html = generate_html_block(st.session_state.news_items)
        st.code(html, language="html")
        st.download_button("Download", data=html, file_name="destacados.html", mime="text/html")
        save_cache(st.session_state.news_items)

# ── Entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if st is None:
        cli_main(sys.argv[1:])
    else:
        gui_main()

# ── Test ───────────────────────────────────────────────────────────────────
if __name__ == "__test__":
    assert classify_article("Opinion: Chile’s carbon markets", "") == "opinion"
    assert classify_article("Gobierno presenta nueva regulación financiera", "") == "econpol"
    assert classify
