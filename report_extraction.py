"""
report_extraction.py
====================

Reads laboratory reports (PDF with a text layer, scanned PDF, or a photo /
scan saved as PNG or JPG) and extracts the six laboratory values used by
RA-Predict:

    HB Level (g/dL), ESR (mm/hr), Creatinine (umol/L),
    CRP (mg/L), Anti-CCP (U/mL), Uric Acid Level (mg/dL)

Units printed on the report are detected and converted to the units used by
the model (for example creatinine in mg/dL is multiplied by 88.4).

Nothing is written to disk. Every extracted value must be checked by the user
against the original report; OCR can misread digits.

Optional dependencies
---------------------
* pypdf          - text PDFs
* pypdfium2      - renders scanned PDFs so they can be read with OCR
* pytesseract    - OCR (also needs the Tesseract program installed)
"""

import io
import re

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover
    Image = None
    ImageOps = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:
    import pypdfium2
except Exception:  # pragma: no cover
    pypdfium2 = None


# ============================================================
# FIELDS
# ============================================================

FILE_TYPES = ("pdf", "png", "jpg", "jpeg")

FIELDS = [
    "HB Level",
    "ESR",
    "Creatinine",
    "CRP",
    "Anti-CCP",
    "Uric Acid Level",
]

FIELD_LABELS = {
    "HB Level": "HB Level (g/dL)",
    "ESR": "ESR (mm/hr)",
    "Creatinine": "Creatinine (µmol/L)",
    "CRP": "CRP (mg/L)",
    "Anti-CCP": "Anti-CCP (U/mL)",
    "Uric Acid Level": "Uric Acid Level (mg/dL)",
}

TARGET_UNITS = {
    "HB Level": "g/dL",
    "ESR": "mm/hr",
    "Creatinine": "µmol/L",
    "CRP": "mg/L",
    "Anti-CCP": "U/mL",
    "Uric Acid Level": "mg/dL",
}

# Names that identify each test on a report
SYNONYMS = {
    "HB Level": r"\b(?:haemoglobin|hemoglobin|hgb|hb)\b(?!\s*a1c)",
    "ESR": (
        r"\besr\b|erythrocyte\s+sedimentation\s+rate|"
        r"\bsedimentation\s+rate\b"
    ),
    "Creatinine": r"\bcreatinine\b|\bcreat\b",
    "CRP": r"\bcrp\b|\bc[\s\-]*reactive\s+protein\b",
    "Anti-CCP": (
        r"\banti[\s\-]*ccp\b|\bccp\b|cyclic\s+citrullinated|\bacpa\b"
    ),
    "Uric Acid Level": r"\buric\s+acid\b|\burate\b",
}

# Lines that contain these words describe a different test
EXCLUDE_LINE = {
    "Creatinine": r"clearance|egfr|\bratio\b|\burine\b|\bgfr\b",
    "HB Level": r"\bmch\b|\bmchc\b|\bhbsag\b",
}

# Values outside these limits are treated as OCR errors and not filled in.
PLAUSIBLE = {
    "HB Level": (3.0, 25.0),
    "ESR": (0.0, 200.0),
    "Creatinine": (15.0, 2000.0),
    "CRP": (0.0, 600.0),
    "Anti-CCP": (0.0, 5000.0),
    "Uric Acid Level": (0.5, 30.0),
}

ROUND_TO = {
    "HB Level": 2,
    "ESR": 1,
    "Creatinine": 1,
    "CRP": 2,
    "Anti-CCP": 2,
    "Uric Acid Level": 2,
}

CREATININE_MGDL_TO_UMOLL = 88.4
URIC_ACID_UMOLL_TO_MGDL = 1.0 / 59.48
URIC_ACID_MMOLL_TO_MGDL = 16.81


# ============================================================
# AVAILABILITY
# ============================================================

def ocr_available():
    """True when Tesseract OCR can be used on this server."""
    if pytesseract is None or Image is None:
        return False

    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ============================================================
# TEXT OF A FILE
# ============================================================

def _ocr_image(image):
    """
    Run OCR on a PIL image with several layout modes and scales.
    Returns a list of texts, the best reading first. The extra readings are
    used to detect digits that OCR may have misread.
    """
    image = ImageOps.exif_transpose(image).convert("L")

    # Small images are upscaled; OCR works best at roughly 300 dpi.
    if image.width < 1600:
        factor = 1600 / image.width
        image = image.resize(
            (int(image.width * factor), int(image.height * factor))
        )

    image = ImageOps.autocontrast(image)

    larger = image.resize((int(image.width * 1.5), int(image.height * 1.5)))

    runs = [(image, "6"), (image, "4"), (larger, "6"), (image, "11")]

    texts = []

    for picture, mode in runs:
        try:
            texts.append(
                pytesseract.image_to_string(picture, config=f"--psm {mode}")
            )
        except Exception:
            continue

    def hits(text):
        return sum(
            1 for pattern in SYNONYMS.values() if re.search(pattern, text, re.I)
        )

    texts.sort(key=hits, reverse=True)

    return texts


def _text_from_pdf(data):
    """Returns (text, method, alternative texts)."""
    text = ""

    if PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(
                (page.extract_text() or "") for page in reader.pages
            )
        except Exception:
            text = ""

    if len(text.strip()) >= 30:
        return text, "PDF text", []

    # No text layer: render pages and use OCR
    if pypdfium2 is not None and ocr_available():
        try:
            pdf = pypdfium2.PdfDocument(data)
            pages = []

            for index in range(min(len(pdf), 5)):
                page_image = pdf[index].render(scale=2.5).to_pil()
                pages.append(_ocr_image(page_image))

            n_runs = min(len(p) for p in pages)

            joined = [
                "\n".join(page[k] for page in pages) for k in range(n_runs)
            ]

            return joined[0], "scanned PDF (OCR)", joined[1:]
        except Exception:
            pass

    return text, None, []


def _text_from_file(name, data):
    """
    Returns (text, method, alternative texts).
    method is None when the file could not be read.
    """
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    if suffix == "pdf":
        return _text_from_pdf(data)

    if suffix in ("png", "jpg", "jpeg"):

        if not ocr_available():
            return "", None, []

        try:
            image = Image.open(io.BytesIO(data))
            texts = _ocr_image(image)
            return texts[0], "image (OCR)", texts[1:]
        except Exception:
            return "", None, []

    return "", None, []


# ============================================================
# VALUE EXTRACTION
# ============================================================

_NUMBER = re.compile(
    r"(?P<qual>[<>≤≥]=?)?\s*(?P<num>\d{1,4}(?:[.,]\d{1,3})?)"
)

_ANY_SYNONYM = re.compile(
    "|".join(f"(?:{p})" for p in SYNONYMS.values()), re.I
)


def _clean_tail(tail):
    """Remove text that is not part of the result."""
    # digits split by OCR letters, e.g. 1O.5 -> 10.5
    tail = re.sub(r"(?<=\d)[Oo](?=\d)", "0", tail)
    tail = re.sub(r"\([^)]*\)", " ", tail)
    tail = re.sub(r"\b\d+\s*(?:st|nd|rd|th)\b", " ", tail, flags=re.I)
    tail = re.sub(r"\b1\s*(?:hr|hour)s?\b", " ", tail, flags=re.I)
    return tail


def _to_float(text):
    return float(text.replace(",", "."))


def _unit_window(tail, num_end):
    """Text just after the number, cut at the next test name."""
    window = tail[num_end:]

    other = _ANY_SYNONYM.search(window)

    if other:
        window = window[: other.start()]

    return window[:32]


def _convert(field, value, window):
    """
    Detect the unit in 'window' and convert 'value' to the model unit.
    Returns (value, unit_text, converted, note).
    """
    w = window.lower().replace("µ", "u").replace("μ", "u")
    w = w.replace(" ", "")

    if field == "HB Level":
        if re.search(r"g/l(?![a-z])", w) and "g/dl" not in w:
            return value / 10.0, "g/L", True, ""
        if re.search(r"g/?dl|gm/?dl|gm%|g%", w):
            return value, "g/dL", False, ""
        return value, "", False, "unit not printed; g/dL assumed"

    if field == "ESR":
        return value, "mm/hr", False, ""

    if field == "Creatinine":
        if "mg/dl" in w or "mg%" in w:
            return value * CREATININE_MGDL_TO_UMOLL, "mg/dL", True, ""
        if re.search(r"umol/l|micromol|umol", w):
            return value, "µmol/L", False, ""
        if value < 20:
            return (
                value * CREATININE_MGDL_TO_UMOLL, "", True,
                "unit not printed; treated as mg/dL because the value "
                "is below 20",
            )
        return value, "", False, "unit not printed; µmol/L assumed"

    if field == "CRP":
        if "mg/dl" in w:
            return value * 10.0, "mg/dL", True, ""
        return value, "mg/L", False, ""

    if field == "Anti-CCP":
        return value, "U/mL", False, ""

    if field == "Uric Acid Level":
        if "mmol/l" in w:
            return value * URIC_ACID_MMOLL_TO_MGDL, "mmol/L", True, ""
        if re.search(r"umol/l|micromol|umol", w):
            return value * URIC_ACID_UMOLL_TO_MGDL, "µmol/L", True, ""
        if "mg/dl" in w or "mg%" in w:
            return value, "mg/dL", False, ""
        if value > 20:
            return (
                value * URIC_ACID_UMOLL_TO_MGDL, "", True,
                "unit not printed; treated as µmol/L because the value "
                "is above 20",
            )
        return value, "", False, "unit not printed; mg/dL assumed"

    return value, "", False, ""


def extract_values(text):
    """
    Find the six laboratory values in a block of text.
    Returns {field: [candidate, ...]} where each candidate is a dict.
    """
    found = {field: [] for field in FIELDS}
    qualitative = {}

    lines = [ln.strip() for ln in text.splitlines()]

    for index, line in enumerate(lines):

        if not line:
            continue

        for field in FIELDS:

            match = re.search(SYNONYMS[field], line, re.I)

            if not match:
                continue

            exclude = EXCLUDE_LINE.get(field)

            if exclude and re.search(exclude, line, re.I):
                continue

            tail = _clean_tail(line[match.end():])
            source = line

            number = _NUMBER.search(tail)

            # Result on the next line (table cells split by OCR)
            if number is None and not re.search(r"[A-Za-z]{2,}", tail):

                for nxt in lines[index + 1: index + 3]:

                    if not nxt:
                        continue

                    nxt_clean = _clean_tail(nxt)

                    candidate = _NUMBER.match(nxt_clean.strip())

                    if candidate:
                        tail, number, source = nxt_clean.strip(), candidate, (
                            line + "  |  " + nxt
                        )

                    break

            if number is None:

                if re.search(
                    r"\b(positive|negative|reactive|non[\s\-]*reactive)\b",
                    line, re.I
                ):
                    qualitative[field] = line

                continue

            raw = _to_float(number.group("num"))

            window = _unit_window(tail, number.end())

            value, unit, converted, note = _convert(field, raw, window)

            qualifier = number.group("qual") or ""

            low, high = PLAUSIBLE[field]

            if not (low <= value <= high):
                found[field].append({
                    "field": field,
                    "rejected": True,
                    "raw": raw,
                    "unit": unit,
                    "source": source,
                    "note": (
                        f"value {raw:g} {unit} is outside the plausible "
                        f"range and was not used"
                    ),
                })
                continue

            if qualifier:
                note = (note + "; " if note else "") + (
                    f"reported as {qualifier}{raw:g}"
                )

            if re.search(r"\bhs[\s\-]*crp\b", line, re.I):
                note = (note + "; " if note else "") + (
                    "high-sensitivity CRP; check it is comparable"
                )

            found[field].append({
                "field": field,
                "rejected": False,
                "value": round(value, ROUND_TO[field]),
                "raw": raw,
                "unit": unit,
                "converted": converted,
                "note": note,
                "source": source,
                "qualifier": qualifier,
            })

    found["_qualitative"] = qualitative

    return found


# ============================================================
# PUBLIC API
# ============================================================

def read_reports(files):
    """
    files: list of (file name, bytes).
    Returns {"files": [(name, method, text)], "found": {field: [candidates]}}.
    Candidates from earlier files come first.
    """
    result_files = []
    merged = {field: [] for field in FIELDS}
    qualitative = {}

    for name, data in files:

        text, method, alternatives = _text_from_file(name, data)

        result_files.append((name, method, text))

        if not text.strip():
            continue

        found = extract_values(text)

        alt_found = [extract_values(t) for t in alternatives if t.strip()]

        for field in FIELDS:

            for candidate in found[field]:

                candidate["file"] = name

                if not candidate.get("rejected"):

                    others = set()

                    for alt in alt_found:

                        usable = [
                            c for c in alt[field] if not c.get("rejected")
                        ]

                        others.add(usable[0]["value"] if usable else None)

                    candidate["readings"] = others

                merged[field].append(candidate)

        qualitative.update(found.get("_qualitative", {}))

    merged["_qualitative"] = qualitative

    return {"files": result_files, "found": merged}


def summarise_for_form(extraction):
    """
    Turns the extraction into what the form needs:
      prefill   {field: value}
      rows      table rows for the user to check
      attention list of warnings
      not_found fields that were not found
      files     [(name, method, text)]
    """
    found = extraction["found"]

    prefill = {}
    rows = []
    attention = []
    not_found = []

    for field in FIELDS:

        usable = [c for c in found[field] if not c.get("rejected")]
        rejected = [c for c in found[field] if c.get("rejected")]

        if not usable:

            not_found.append(FIELD_LABELS[field])

            for c in rejected:
                attention.append(f"{FIELD_LABELS[field]}: {c['note']}.")

            if field in found.get("_qualitative", {}):
                attention.append(
                    f"{FIELD_LABELS[field]}: the report gives a "
                    f"qualitative result only "
                    f"({found['_qualitative'][field][:60]})."
                )

            continue

        chosen = usable[0]

        prefill[field] = chosen["value"]

        distinct = {c["value"] for c in usable}

        if len(distinct) > 1:
            attention.append(
                f"{FIELD_LABELS[field]}: more than one value was found "
                f"({', '.join(f'{v:g}' for v in sorted(distinct))}); the "
                f"first one was used."
            )

        if chosen["note"]:
            attention.append(f"{FIELD_LABELS[field]}: {chosen['note']}.")

        readings = chosen.get("readings") or set()

        disagree = {r for r in readings if r != chosen["value"]}

        if disagree:
            shown = ", ".join(
                "not read" if r is None else f"{r:g}" for r in sorted(
                    disagree, key=lambda v: (v is None, v or 0)
                )
            )
            attention.append(
                f"{FIELD_LABELS[field]}: the text recognition gave "
                f"different results on repeated reading (used "
                f"{chosen['value']:g}; other readings: {shown}). "
                f"Check this value carefully."
            )

        printed = f"{chosen['raw']:g} {chosen['unit']}".strip()

        rows.append({
            "Variable": FIELD_LABELS[field],
            "Value filled": chosen["value"],
            "As printed on the report": printed,
            "Converted": "yes" if chosen["converted"] else "no",
            "Check": "readings differ" if disagree else "",
            "Source line": chosen["source"][:80],
            "File": chosen.get("file", ""),
        })

    return {
        "prefill": prefill,
        "rows": rows,
        "attention": attention,
        "not_found": not_found,
        "files": extraction["files"],
    }
