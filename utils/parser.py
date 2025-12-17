import re
import unicodedata
from difflib import get_close_matches


# ------------------------------------------------------
# Basic text normalization helper
# ------------------------------------------------------
def normalize_text(s: str):
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf8")
    s = s.replace("mahallesi", "").replace("mah", "").replace(".", " ")
    return " ".join(s.split())


# ------------------------------------------------------
# Regex fallback mahalle parser
# ------------------------------------------------------
def parse_mahalle_regex(addr):
    p = r"\b([A-ZÇĞİÖŞÜa-zçğıöşü0-9 \-']+?)\s*(MAHALLES[İI]?|MAHALLE|MAH\.?|MH\.?|MHL\.?)\b"
    m = re.search(p, addr, re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ------------------------------------------------------
# Cadde parser
# ------------------------------------------------------
def parse_cadde(addr):
    p = r"\b([A-ZÇĞİÖŞÜa-zçğıöşü0-9 \-']{3,})\s*(CADDE?|CADDES[İI]?|CAD\.?|CD\.?)\b"
    m = re.search(p, addr, re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ------------------------------------------------------
# Sokak parser
# ------------------------------------------------------
def parse_sokak(addr):
    p = r"\b([A-ZÇĞİÖŞÜa-zçğıöşü0-9][A-ZÇĞİÖŞÜa-zçğıöşü0-9 \-']{2,})\s*" \
        r"(SOKAĞ[İI]?|SOKAG[İI]?|SOKAK|SOK\.?|SK\.?|SK)\b"

    m = re.search(p, addr, re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ------------------------------------------------------
# SMART MAHALLE DETECTOR (Uses your Excel mahalle list)
# ------------------------------------------------------
def smart_mahalle_detector(addr: str, il: str, ilce: str, mahalle_df):
    """Use mahalle DB + fuzzy matching + regex fallback."""
    norm_addr = normalize_text(addr)

    # Filter mahalle list for IL & ILCE
    subset = mahalle_df[
        (mahalle_df["il"] == il.lower()) &
        (mahalle_df["ilçe"] == ilce.lower())
    ]

    if subset.empty:
        return parse_mahalle_regex(addr)

    mahalle_list = list(subset["mahalle"].unique())
    normalized_list = [normalize_text(m) for m in mahalle_list]

    # Direct containment match
    for m_norm, m_orig in zip(normalized_list, mahalle_list):
        if m_norm in norm_addr:
            return m_orig

    # Fuzzy fallback
    match = get_close_matches(norm_addr, normalized_list, n=1, cutoff=0.80)
    if match:
        idx = normalized_list.index(match[0])
        return mahalle_list[idx]

    return parse_mahalle_regex(addr)
