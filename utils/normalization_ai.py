import openai
import os
import re
import unicodedata
import streamlit as st

# Load separate key for normalization only
NORMALIZER_KEY = os.getenv("OPENAI_NORMALIZER_KEY") or st.secrets.get("OPENAI_NORMALIZER_KEY")
openai.api_key = NORMALIZER_KEY


SYSTEM_PROMPT = """
You are an advanced Turkish address normalization engine.
Normalize Turkish addresses while preserving:
- mahalle
- cadde / sokak
- no, daire, kat, blok
Fix spelling, casing, spacing, punctuation, and Turkish characters.
Expand abbreviations (mah → Mahallesi, sk → Sokak, cad → Cadde).
Do NOT invent missing data.
Return a SINGLE normalized address string.
"""


def ascii_fallback(text: str) -> str:
    """Convert Turkish characters to ASCII safe equivalents."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def ai_normalize_address(addr: str) -> str:
    """Normalize Turkish address with OpenAI, then ASCII-normalize."""
    
    # No key? ASCII fallback only
    if not NORMALIZER_KEY:
        return ascii_fallback(addr)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": addr},
            ],
            max_tokens=180,
            temperature=0,
        )
        cleaned = response["choices"][0]["message"]["content"].strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        # FINAL ASCII normalization for geocoding compatibility
        return ascii_fallback(cleaned)

    except Exception:
        return ascii_fallback(addr)
