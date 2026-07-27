"""Arachne Scholar -- PDF -> Markdown (OCR diretto via Ollama, ZERO SDK).

Architettura minimale (post-SDK, 2026-07-27):
  - Niente SDK glmocr, niente config yaml, niente layout detector esterno,
    niente venv separati. Il backend parla con Ollama via HTTP stdlib.
  - Ogni pagina PDF viene rasterizzata IN MEMORIA da PyMuPDF (niente poppler,
    niente pdf2image: PyMuPDF e' gia' dipendenza core e produce PNG nativi).
  - L'immagine va a POST {OLLAMA_BASE_URL}/api/generate con il prompt
    ufficiale "Text Recognition:" (unico prompt sensato a pagina intera:
    Table/Formula Recognition esistono solo per regioni croppate dal layout
    detector, che qui non esiste piu' per scelta architetturale).
  - Contesto 16k FORZATO PER RICHIESTA via options.num_ctx=16384: una pagina
    vale ~6k token visivi + output lungo; col default 4096 il context-shift
    troncava e mandava il modello in loop (bug verificato in produzione).
  - keep_alive 30m: il modello resta in VRAM tra una pagina e l'altra.
  - Fallback a due livelli, mai bloccante:
      * pagina fallita  -> testo nativo PyMuPDF per QUELLA pagina
      * Ollama giu'     -> intero PDF al percorso classico (TIER 3)

Override via settings.json (repo root) o variabili d'ambiente:
  ocr_mode: "auto" (default) | "ollama" (forza OCR) | "classic"
  ollama_base_url / OLLAMA_BASE_URL:  default http://localhost:11434
  ollama_model / OLLAMA_OCR_MODEL:    default glm-ocr-16k (auto-risoluzione
                                      sui tag disponibili: glm-ocr-16k ->
                                      glm-ocr:latest -> glm-ocr)
  ocr_page_timeout:                   secondi max per pagina (default 600)
  ocr_dpi:                            risoluzione raster (default 200)

Il parser SVO di spaCy NON viene toccato qui: questo modulo produce solo
Markdown pulito in data/converted_md/.
"""
import os, sys, glob, re, json, time, base64
import urllib.request
from collections import Counter

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Errore: PyMuPDF non installato. Esegui 'pip install pymupdf'")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from hardware_probe import probe_hardware
except Exception:
    probe_hardware = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OCR_MODEL = "glm-ocr-16k"
# GLM-OCR accetta SOLO 3 prompt fissi (Text/Table/Formula Recognition).
# A pagina intera si usa sempre e solo questo:
OCR_PROMPT = "Text Recognition:"
# Generazione deterministica + finestra 16k (vedi docstring modulo).
OCR_OPTIONS = {
    "num_ctx": 16384,
    "num_predict": 8192,
    "temperature": 0.0,
    "top_p": 0.00001,
    "top_k": 1,
    "repetition_penalty": 1.1,
}


def load_settings():
    try:
        return json.load(open(SETTINGS_FILE))
    except Exception:
        return {}


def strip_boilerplate(text):
    """Rimuove URL, DOI, ISBN, copyright e rumore editoriale dal testo
    grezzo PRIMA della conversione markdown — evita che spaCy tratti
    licenze e link come entita' fittizie ad alta occorrenza."""
    text = re.sub(r'https?://[^\s)>\]"\'»«]+', ' ', text)
    text = re.sub(r'\b10\.\d{4,}[^\s>\]"\'»«]*', ' ', text)
    text = re.sub(r'\b(?:ISBN\s*:?\s*)?(?:97[89][- ]?)?\d{1,5}[- ]\d{1,7}[- ]\d{1,7}[- ][\dX]\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'©\s*\d{4}\s+.*?(?:\n|\.)', '\n', text)
    text = re.sub(r'(?:Creative Commons|CC\s+BY)[^\n]{0,200}(?:\n|License)', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'All\s+rights\s+reserved\.?', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', ' ', text)
    text = re.sub(r'^(?:This\s+(?:work|article|book|publication|eBook)\s+is\s+(?:published|made\s+available|copyrighted|distributed|licensed)).*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'(?:Published\s+by|Digital\s+Object\s+Identifier|Licensed\s+under).*', '', text, flags=re.IGNORECASE)
    return text


# --------------------------------------------------------------------- TIER 3
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
              "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"}


def sanitize_markdown(text, n_pages=1):
    """Sanitizzazione regex pesante per l'estrazione testuale classica.
    Ripara gli artefatti tipici del PDF-to-text che inquinano spaCy:
    sillabazioni a fine riga, legature, apici di nota, numeri di pagina,
    header/footer ripetuti. Ritorna (testo_pulito, stats)."""
    stats = Counter()

    for lig, rep in _LIGATURES.items():
        n = text.count(lig)
        if n:
            stats["legature"] += n
            text = text.replace(lig, rep)

    n = text.count("\u00ad")
    if n:
        stats["soft_hyphen"] += n
        text = text.replace("\u00ad", "")

    text, n = re.subn(r'(?<=[a-zàèéìòùç])[‐‑-]\n(?=[a-zàèéìòùç])', '', text)
    stats["sillabazioni_riparate"] += n

    text, n = re.subn(r'(?<=[\w\)\.])(?:\[\d{1,3}(?:[,\u2013-]\d{1,3})*\])', '', text)
    stats["note_apice_quadre"] += n

    text, n = re.subn(r'(?<=[a-zàèéìòù])\.?\d{1,2}(?=\s+[A-ZÀÈÉÌÒÙ(“"])', '', text)
    stats["note_apice_numeriche"] += n

    lines = text.split("\n")
    counts = Counter(l.strip() for l in lines if l.strip())
    thresh = max(3, int(0.3 * max(n_pages, 1)))
    kept = []
    for l in lines:
        s = l.strip()
        if s and re.fullmatch(r'(\d{1,4}|[ivxlcdmIVXLCDM]{1,7})', s):
            stats["numeri_pagina"] += 1
            continue
        if s and len(s) < 80 and counts[s] >= thresh:
            stats["header_footer_ripetuti"] += 1
            continue
        kept.append(l)
    text = "\n".join(kept)

    return text, stats


def _light_cleanup(text):
    """Pulizia leggera per l'output OCR (markdown strutturato):
    niente de-sillabazione ne' rimozione righe — le tabelle markdown
    verrebbero danneggiate. Solo legature + whitespace."""
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = text.replace("\u00ad", "")
    return re.sub(r'\n{3,}', '\n\n', text)


# ------------------------------------------------------------- Ollama direct
def _http_json(url, payload=None, timeout=30):
    """POST (se payload) o GET JSON via SOLA stdlib. Solleva eccezione."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def resolve_ollama_model(base_url, wanted):
    """Ritorna il nome modello realmente disponibile su Ollama, o None se il
    server e' giu' o non ha nessun glm-ocr. Ordine: wanted -> glm-ocr-16k ->
    glm-ocr:latest -> glm-ocr (portabilita': basta `ollama pull glm-ocr`)."""
    try:
        tags = _http_json(base_url + "/api/tags", timeout=4)
        names = [m.get("name", "") for m in tags.get("models", [])]
    except Exception:
        return None
    candidates = [wanted]
    if ":" not in wanted:
        candidates.append(wanted + ":latest")
    candidates += ["glm-ocr-16k", "glm-ocr:latest", "glm-ocr"]
    for c in candidates:
        if c in names:
            return c
    return None


def ocr_page_ollama(png_bytes, model, base_url, timeout):
    """Una pagina -> markdown via POST /api/generate. Ritorna None se fallita."""
    payload = {
        "model": model,
        "prompt": OCR_PROMPT,
        "images": [base64.b64encode(png_bytes).decode("ascii")],
        "stream": False,
        "keep_alive": "30m",
        "options": OCR_OPTIONS,
    }
    try:
        r = _http_json(base_url + "/api/generate", payload, timeout=timeout)
        return (r.get("response") or "").strip() or None
    except Exception as e:
        print(f"[ollama-ocr] richiesta fallita: {type(e).__name__}: {e}")
        return None


def convert_pdf_ollama(pdf_path, out_dir, settings):
    """OCR diretto pagina-per-pagina via Ollama. Ritorna False solo se il
    server/modello non e' raggiungibile (il chiamante fa fallback classico);
    le singole pagine fallite degradano al testo nativo di QUELLA pagina."""
    base_url = (os.environ.get("OLLAMA_BASE_URL")
                or settings.get("ollama_base_url")
                or DEFAULT_OLLAMA_URL).rstrip("/")
    wanted = (os.environ.get("OLLAMA_OCR_MODEL")
              or settings.get("ollama_model") or DEFAULT_OCR_MODEL)
    model = resolve_ollama_model(base_url, wanted)
    if not model:
        print(f"[ollama-ocr] {base_url} irraggiungibile o nessun modello "
              f"glm-ocr nei tag -> fallback PyMuPDF")
        return False
    dpi = int(settings.get("ocr_dpi", 200))
    timeout = int(settings.get("ocr_page_timeout", 600))
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)

    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    pages_md, n_native = [], 0
    for i in range(1, n_pages + 1):
        page = doc.load_page(i - 1)
        png = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB).tobytes("png")
        t0 = time.time()
        text = ocr_page_ollama(png, model, base_url, timeout)
        if text:
            pages_md.append(text)
            print(f"[ollama-ocr] pagina {i}/{n_pages} ok "
                  f"({time.time() - t0:.0f}s, {len(text)} char)")
        else:
            n_native += 1
            pages_md.append(page.get_text("text"))
            print(f"[ollama-ocr] pagina {i}/{n_pages} FALLITA -> testo "
                  f"nativo PyMuPDF per questa pagina")
    doc.close()

    full = _light_cleanup(strip_boilerplate("\n\n".join(pages_md)))
    base_name = os.path.basename(pdf_path).replace(".pdf", ".md")
    out_path = os.path.join(out_dir, base_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full)
    print(f"Convertito (Ollama OCR {model}@{base_url}, dpi={dpi}, "
          f"pagine-native={n_native}): {base_name}")
    return True


# ------------------------------------------------------------------- TIER 3
def convert_pdf_classic(pdf_path, out_dir):
    """TIER 3: estrazione PyMuPDF + boilerplate filter + sanitize pesante."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    n_pages = doc.page_count
    doc.close()
    text = strip_boilerplate(text)
    text, stats = sanitize_markdown(text, n_pages=n_pages)
    text = re.sub(r'\n{3,}', '\n\n', text)
    base_name = os.path.basename(pdf_path).replace(".pdf", ".md")
    out_path = os.path.join(out_dir, base_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Convertito (PyMuPDF+sanitize): {base_name} | sanitize: {dict(stats)}")
    return True


def convert_pdf_to_md(pdf_path, out_dir, hw=None, settings=None):
    """Router. Priorita' modalita': env ARACHNE_OCR_MODE > settings.ocr_mode
    > auto. auto/ollama: tenta OCR diretto, poi fallback classico per quel PDF.
    Il parametro hw e' accettato per retro-compatibilita' e ignorato."""
    settings = settings if settings is not None else load_settings()
    mode = (os.environ.get("ARACHNE_OCR_MODE")
            or settings.get("ocr_mode") or "auto").lower()
    if mode == "classic":
        return convert_pdf_classic(pdf_path, out_dir)
    if not convert_pdf_ollama(pdf_path, out_dir, settings):
        return convert_pdf_classic(pdf_path, out_dir)
    return True


if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/raw_pdfs"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "../data/converted_md"
    if len(sys.argv) > 3:
        os.environ["ARACHNE_OCR_MODE"] = sys.argv[3]  # override CLI opzionale
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    hw = probe_hardware() if probe_hardware else {}
    print(f"[hardware] {hw.get('tier_label', 'probe assente')} | "
          f"GPU: {hw.get('gpu_name')} | "
          f"VRAM: {hw.get('vram_used_mb')}/{hw.get('vram_total_mb')} MB | "
          f"OCR: {hw.get('ocr_model') or 'non disponibile'}")

    pdfs = glob.glob(os.path.join(in_dir, "*.pdf"))
    if not pdfs:
        print(f"Nessun PDF trovato in {in_dir}")
    else:
        for pdf in pdfs:
            convert_pdf_to_md(pdf, out_dir)
