"""
Marcia News Curator
===================

One‑file tool to curate *Destacados de la Semana* with either a Streamlit GUI or a CLI.

Usage
-----
GUI (if **streamlit** is installed)
```
streamlit run marcia_streamlit_app.py
```
CLI (works everywhere)
```
python marcia_streamlit_app.py --url https://example.com
python marcia_streamlit_app.py --img screenshot.png
python marcia_streamlit_app.py --generate-html
```
A local `.marcia_cache.json` stores up to ten curated items.
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
from typing import Any, Dict, List

try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    st = None

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

    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    OPENAI_AVAILABLE = bool(openai.api_key)
except ImportError:
    OPENAI_AVAILABLE = False

TEMPLATE = (
    "<div class=\"news-item\">\n"
    "  <a class=\"news-link\" href=\"{url}\">{title}</a>\n"
    "  <div class=\"news-details\">{source} — {date}</div>\n"
    "</div>"
)

GROUP_ORDER = {"econpol": 0, "coyuntural": 1, "opinion": 2}


def classify_article(title: str, text: str) -> str:
    if OPENAI_AVAILABLE:
        prompt = (
            "Classify the following climate‑related news as ECONPOL, COYUNTURAL or OPINION.\n\n"
            f"Headline: {title}\nBody: {text[:500]}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
            )
            label = response.choices[0].message.content.strip().upper()
            if label in ("ECONPOL", "COYUNTURAL", "OPINION"):
                return label.lower()
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


def extract_from_image(src: Any) -> Dict[str, Any]:
    image = _open_image(src)
    text = pytesseract.image_to_string(image, lang="spa+eng")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0] if lines else "Untitled"
    date = next((d for l in lines if (d := guess_date(l))), None) or datetime.today().date()
    return {"title": title, "source": "Desconocido", "date": date, "url": "", "raw_text": text}


def extract_from_url(url: str):
    if Article is None:
        return None
    try:
        article = Article(url, language="es")
        article.download()
        article.parse()
        title = article.title or "Untitled"
        text = article.text
        soup = BeautifulSoup(article.html, "html.parser") if BeautifulSoup else None
        src = (
            soup.find("meta", property="og:site_name").get("content")  # type: ignore[attr-defined]
            if soup and soup.find("meta", property="og:site_name") else url.split("/")[2]
        )
        date_meta = soup.find("meta", property="article:published_time") if soup else None  # type: ignore[attr-defined]
        pub_date = guess_date(date_meta.get("content") if date_meta else None) or guess_date(article.publish_date) or datetime.today().date()
        return {"title": title, "source": src, "date": pub_date, "url": url, "raw_text": text}
    except Exception:
        return None


CACHE_FILE = Path(".marcia_cache.json")


def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return []


def save_cache(data):
    CACHE_FILE.write_text(json.dumps(data, default=str, ensure_ascii=False, indent=2))


def add_item(storage, item):
    item["group"] = classify_article(item["title"], item.get("raw_text", ""))
    storage.append(item)
    storage.sort(key=lambda n: (GROUP_ORDER.get(n["group"], 99), -n["date"].toordinal()))
    del storage[10:]


def generate_html_block(storage):
    ordered = sorted(storage, key=lambda n: (GROUP_ORDER[n["group"]], -n["date"].toordinal()))
    return "\n".join(
        TEMPLATE.format(url=n["url"] or "#", title=n["title"], source=n["source"], date=n["date"].strftime("%d %b %Y"))
        for n in ordered
    )


def cli_main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--url")
    p.add_argument("--img")
    p.add_argument("--generate-html", action="store_true")
    args = p.parse_args(argv)
    items = load_cache()
    if args.url:
        it = extract_from_url(args.url)
        if it:
            add_item(items, it)
    if args.img:
        it = extract_from_image(Path(args.img))
        add_item(items, it)
    save_cache(items)
    if args.generate_html:
        print(generate_html_block(items))


def gui_main():
    st.set_page_config(page_title="Marcia – Climate News Curator", layout="wide")
    st.title("📰 Marcia – Climate News Curator")
    if "news_items" not in st.session_state:
        st.session_state.news_items = load_cache()
    with st.sidebar:
        upload = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "pdf"])
        url_input = st.text_input("Paste URL")
        if st.button("Add Image") and upload is not None:
            add_item(st.session_state.news_items, extract_from_image(upload.getvalue()))
        if st.button("Add URL") and url_input:
            it = extract_from_url(url_input)
            if it:
                add_item(st.session_state.news_items, it)
    df = [{k: (v.strftime("%Y-%m-%d") if k == "date" else v) for k, v in n.items() if k != "raw_text"} for n in st.session_state.news_items]
    if df:
        edited = st.data_editor(df, num_rows="dynamic")
        st.session_state.news_items = [
            {
                "title": r["title"],
                "source": r["source"],
                "date": guess_date(r["date"]) or datetime.today().date(),
                "group": r["group"],
                "url": r["url"],
                "raw_text": "",
            }
            for r in edited
        ]
    if st.button("Generate HTML"):
        html = generate_html_block(st.session_state.news_items)
        st.code(html, language="html")
        st.download_button("Download", data=html, file_name="destacados.html", mime="text/html")
        save_cache(st.session_state.news_items)


if __name__ == "__main__":
    if st is None:
        cli_main(sys.argv[1:])
    else:
        gui_main()

if __name__ == "__test__":
    assert classify_article("Opinion: Chile’s carbon markets", "") == "opinion"
    assert classify_article("Gobierno presenta nueva regulación financiera", "") == "econpol"
    assert classify_article("Se inaugura la COP30", "") == "coyuntural"
