"""
Marcia News Curator – stable release
===================================

Single‑file **Streamlit GUI** + **CLI** that collects climate‑related news from
URLs (scraping) or images (OCR) and generates an HTML block ready for
Fidelizador.  Designed to run locally *and* on Streamlit Community Cloud.

Quick start (GUI)
-----------------
```bash
streamlit run marcia_streamlit_app.py
```

Quick start (CLI)
-----------------
```bash
python marcia_streamlit_app.py --url https://example.com/news
python marcia_streamlit_app.py --img headline.png
python marcia_streamlit_app.py --generate-html  # prints current block
```

requirements.txt
----------------
```
streamlit
requests
beautifulsoup4
requests_cache
lxml
cssselect
python-dateutil
Pillow          # local image handling
pytesseract     # local OCR (optional)
openai          # GPT classification (optional)
newspaper3k     # better scraping (optional)
```

Secrets (Streamlit Cloud)
------------------------
```toml
OPENAI_API_KEY = "sk‑…"            # optional
OCR_SPACE_API_KEY = "your_ocr_key"  # optional – enables online OCR
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

# ───────────────────────────────────────────────────────────────────────────
# Streamlit detection (GUI vs CLI)                                           
# ───────────────────────────────────────────────────────────────────────────
try:
    import streamlit as st  # type: ignore
except ModuleNotFoundError:
    st = None  # noqa: N816 – fallback to CLI only

# ───────────────────────────────────────────────────────────────────────────
# Optional OCR: Tesseract (local) + OCR.space (online)                        
# ───────────────────────────────────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image
    try:
        pytesseract.get_tesseract_version()
        _LOCAL_OCR = True
    except (pytesseract.TesseractNotFoundError, RuntimeError):
        _LOCAL_OCR = False
except ImportError:
    _LOCAL_OCR = False

_OCR_API_KEY = os.getenv("OCR_SPACE_API_KEY") or (
    st.secrets.get("OCR_SPACE_API_KEY", "") if st and hasattr(st, "secrets") else ""
)
_ONLINE_OCR = bool(_OCR_API_KEY)
OCR_ENABLED = _LOCAL_OCR or _ONLINE_OCR

# ───────────────────────────────────────────────────────────────────────────
# HTTP + scraping                                                            
# ───────────────────────────────────────────────────────────────────────────
try:
    import requests
    import requests_cache
    requests_cache.install_cache("marcia_cache", expire_after=3_600)
except ImportError:
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    BeautifulSoup = None

try:
    from newspaper import Article  # type: ignore
except ImportError:
    Article = None

from dateutil import parser as dateparser

# ───────────────────────────────────────────────────────────────────────────
# GPT classification (optional)                                              
# ───────────────────────────────────────────────────────────────────────────
try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

_OPENAI_KEY = os.getenv("OPENAI_API_KEY") or (
    st.secrets.get("OPENAI_API_KEY", "") if st and hasattr(st, "secrets") else ""
)
GPT_ENABLED = bool(openai and _OPENAI_KEY)
if GPT_ENABLED:
    openai.api_key = _OPENAI_KEY

# ───────────────────────────────────────────────────────────────────────────
# Constants                                                                  
# ───────────────────────────────────────────────────────────────────────────
TEMPLATE = (
    "<div class=\"news-item\">\n"
    "  <a class=\"news-link\" href=\"{url}\">{title}</a>\n"
    "  <div class=\"news-details\">{source} — {date}</div>\n"
    "</div>"
)
ORDER = {"econpol": 0, "coyuntural": 1, "opinion": 2}
CACHE = Path(".marcia_cache.json")

# ───────────────────────────────────────────────────────────────────────────
# Helper functions                                                           
# ───────────────────────────────────────────────────────────────────────────

def classify(title: str, body: str = "") -> str:
    """Return group label for the headline."""
    if GPT_ENABLED:
        prompt = (
            "Classify the following climate‑related headline as ECONPOL, COYUNTURAL or OPINION."\
            f"\nHeadline: {title}\nBody: {body[:400]}"
        )
        try:
            rsp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
            )
            lbl = rsp.choices[0].message.content.strip().upper()
            if lbl in ("ECONPOL", "COYUNTURAL", "OPINION"):
                return lbl.lower()
        except Exception:
            pass
    # fallback keywords
    if re.search(r"\b(opinion|opinión|editorial|column)\b", title, re.I):
        return "opinion"
    if re.search(r"\b(finanzas|mercado|banco|gobierno|regulación|policy|impuesto|economía)\b", title, re.I):
        return "econpol"
    return "coyuntural"


def parse_date(text: str | None):
    if not text:
        return None
    try:
        return dateparser.parse(text, dayfirst=False).date()
    except Exception:
        return None

# ───────────────────────────────────────────────────────────────────────────
# OCR routines                                                               
# ───────────────────────────────────────────────────────────────────────────

def _open_img(src: Any):
    from PIL import Image  # local import
    if isinstance(src, (bytes, bytearray)):
        return Image.open(BytesIO(src))
    if hasattr(src, "read"):
        return Image.open(src)
    return Image.open(Path(src))


def _ocr_space(img_bytes: bytes):
    if not (_ONLINE_OCR and requests):
        return None, "OCR online no disponible."
    try:
        r = requests.post(
            "https://api.ocr.space/parse/image",
            headers={"apikey": _OCR_API_KEY},
            files={"file": ("img.png", img_bytes)},
            data={"language": "spa", "isOverlayRequired": False},
            timeout=30,
        )
        r.raise_for_status()
        text = r.json().get("ParsedResults", [{}])[0].get("ParsedText", "")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None, "OCR online no detectó texto."
        title = lines[0]
        date = next((d for l in lines if (d := parse_date(l))), None) or datetime.today().date()
        return {"title": title, "source": "Desconocido", "date": date, "url": "", "raw_text": text}, ""
    except Exception as e:
        return None, f"OCR online error: {e}"


def extract_image(src: Any):
    """Return (item, error_msg) from image."""
    # Try local OCR first
    if _LOCAL_OCR:
        try:
            img = _open_img(src)
            txt = pytesseract.image_to_string(img, lang="spa+eng")
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            if not lines:
                return None, "No se encontró texto legible en la imagen."
            title = lines[0]
            date = next((d for l in lines if (d := parse_date(l))), None) or datetime.today().date()
            return {"title": title, "source": "Desconocido", "date": date, "url": "", "raw_text": txt}, ""
        except Exception as e:
            return None, f"OCR local error: {e}"
    # Online fallback
    try:
        bts = src if isinstance(src, (bytes, bytearray)) else _open_img(src).tobytes()
    except Exception as e:
        return None, f"Error leyendo imagen: {e}"
    return _ocr_space(bts)

# ───────────────────────────────────────────────────────────────────────────
# Scraping routines                                                          
# ───────────────────────────────────────────────────────────────────────────

def _scrape_requests(url: str):
    if not (requests and BeautifulSoup):
        return None, "Dependencias de scraping faltantes."
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title_tag = soup.find("meta", property="og:title") or soup.find("title")
        title = title_tag.get("content") if title_tag and title_tag.get("content") else title_tag.get_text(strip=True) if title_tag else "Untitled"
        body = soup.get_text(" ", strip=True)[:2000]
        site_tag = soup.find("meta", property="og:site_name")
        source = site_tag.get("content") if site_tag and site_tag.get("content") else url.split("/")[2]
        date_meta = soup.find("meta", property="article:published_time")
        pub_date = parse_date(date_meta.get("content") if date_meta and date_meta.get("content") else None) or datetime.today().date()
        return {"title": title, "source": source, "date": pub_date, "url": url, "raw_text": body}, ""
    except Exception as e:
        return None, f"Scraping error: {e}"


def extract_url(url: str):
    if Article:
        try:
            art = Article(url, language="es"); art.download(); art.parse()
            title = art.title or "Untitled"
            text = art.text or ""
            source = url.split("/")[2]
            pub_date = parse_date(str(art.publish_date)) or datetime.today().date()
            return {"title": title, "source": source, "date": pub_date, "url": url, "raw_text": text}, ""
        except Exception:
            pass
    # fallback
    return _scrape_requests(url)

# ───────────────────────────────────────────────────────────────────────────
# Persistence                                                               
# ───────────────────────────────────────────────────────────────────────────

def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return []


def save_cache(data):
    CACHE.write_text(json.dumps(data, default=str, ensure_ascii=False, indent=2))

# ───────────────────────────────────────────────────────────────────────────
# Core                                                                       
# ───────────────────────────────────────────────────────────────────────────

def add_item(store: List[Dict[str, Any]], item: Dict[str, Any]):
    item["group"] = classify(item["title"], item.get("raw_text", ""))
    store.append(item)
    store.sort(key=lambda n: (ORDER[n["group"]], -n["date"].toordinal()))
    del store
