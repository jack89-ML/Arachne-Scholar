"""Arachne Scholar -- PDF -> Markdown (pre-processor hardware-aware).

Switch dinamico a 3 tier (diagnostica da src/hardware_probe.py):

  TIER 1  VRAM > 12GB  -> GLM-OCR completo su GPU (layout + riconoscimento)
  TIER 2  VRAM <= 12GB -> GLM-OCR ibrido: --layout-device cpu, OCR su GPU
                          (backend VLM via Ollama: non tocca il venv Arachne)
  TIER 3  no GPU / <3GB-> PyMuPDF classico + SANITIZZAZIONE REGEX pesante

Override via settings.json (repo root) o variabili d'ambiente:
  ocr_mode: "auto" (default) | "glm" (forza GLM-OCR) | "classic"
  glmocr_bin / GLMOCR_BIN:       path al binario glmocr (venv dedicato)
  glmocr_config / GLMOCR_CONFIG: config.yaml dell'SDK (backend Ollama)
  glmocr_timeout:                secondi max per singolo PDF (default 1800)

Il parser SVO di spaCy NON viene toccato qui: questo modulo produce solo
Markdown pulito in data/converted_md/.
"""
import os, sys, glob, re, json, shutil, subprocess, tempfile
from collections import Counter

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Errore: PyMuPDF non installato. Esegui 'pip install pymupdf'")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from hardware_probe import probe_hardware, TIER_LAYOUT_DEVICE, TIER_LABELS
except Exception:
    probe_hardware = None
    TIER_LAYOUT_DEVICE = {1: "cuda:0", 2: "cpu", 3: None}
    TIER_LABELS = {1: "TIER 1 - GLM-OCR full-GPU", 2: "TIER 2 - GLM-OCR ibrido",
                   3: "TIER 3 - PyMuPDF + sanitizzazione regex"}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
GLMOCR_CONFIG_PATH = os.path.join(BASE_DIR, "glmocr_config.yaml")


def load_settings():
    try:
        return json.load(open(SETTINGS_FILE))
    except Exception:
        return {}


def strip_boilerplate(text):
    """(FIX 1) Rimuove URL, DOI, ISBN, copyright e rumore editoriale dal testo
    grezzo del PDF PRIMA della conversione markdown — evita che spaCy tratti
    licenze e link come entita' fittizie ad alta occorrenza."""
    # URL
    text = re.sub(r'https?://[^\s)>\]"\'»«]+', ' ', text)
    # DOI
    text = re.sub(r'\b10\.\d{4,}[^\s>\]"\'»«]*', ' ', text)
    # ISBN
    text = re.sub(r'\b(?:ISBN\s*:?\s*)?(?:97[89][- ]?)?\d{1,5}[- ]\d{1,7}[- ]\d{1,7}[- ][\dX]\b', ' ', text, flags=re.IGNORECASE)
    # Copyright / Creative Commons / publisher boilerplate
    text = re.sub(r'©\s*\d{4}\s+.*?(?:\n|\.)', '\n', text)
    text = re.sub(r'(?:Creative Commons|CC\s+BY)[^\n]{0,200}(?:\n|License)', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'All\s+rights\s+reserved\.?', '\n', text, flags=re.IGNORECASE)
    # email
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b', ' ', text)
    # accesso / licenze (Springer, Elsevier, PubMed, etc.)
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

    # 1) Legature tipografiche (ﬁ ﬂ ﬀ ...) -> ASCII: spaCy non le conosce
    for lig, rep in _LIGATURES.items():
        n = text.count(lig)
        if n:
            stats["legature"] += n
            text = text.replace(lig, rep)

    # 2) Soft hyphen invisibili (U+00AD)
    n = text.count("\u00ad")
    if n:
        stats["soft_hyphen"] += n
        text = text.replace("\u00ad", "")

    # 3) De-sillabazione a fine riga: "informa-\ntione" -> "informazione".
    #    Solo tra minuscole: riduce i falsi positivi su parole composte.
    text, n = re.subn(r'(?<=[a-zàèéìòùç])[‐‑-]\n(?=[a-zàèéìòùç])', '', text)
    stats["sillabazioni_riparate"] += n

    # 4) Rimandi a note in stile [12] / [1,2] / [3-5] attaccati al testo
    text, n = re.subn(r'(?<=[\w\)\.])(?:\[\d{1,3}(?:[,\u2013-]\d{1,3})*\])', '', text)
    stats["note_apice_quadre"] += n

    # 5) Apici numerici incollati: "parola12 La" / "frase.3 Il" (1-2 cifre,
    #    seguiti da spazio + maiuscola: pattern tipico della nota a pie' pagina)
    text, n = re.subn(r'(?<=[a-zàèéìòù])\.?\d{1,2}(?=\s+[A-ZÀÈÉÌÒÙ(“"])', '', text)
    stats["note_apice_numeriche"] += n

    # 6) Furniture di pagina: numeri di pagina isolati e header/footer
    #    ripetuti (stessa riga su >= 30% delle pagine, min 3 occorrenze)
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
    """Pulizia leggera per l'output GLM-OCR (markdown strutturato):
    niente de-sillabazione ne' rimozione righe — le tabelle markdown
    verrebbero danneggiate. Solo legature + whitespace."""
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = text.replace("\u00ad", "")
    return re.sub(r'\n{3,}', '\n\n', text)


# ------------------------------------------------------------------- backends
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
    text = re.sub(r'\n{3,}', '\n\n', text)  # collassa whitespace eccessivo
    base_name = os.path.basename(pdf_path).replace(".pdf", ".md")
    out_path = os.path.join(out_dir, base_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Convertito (PyMuPDF+sanitize): {base_name} | sanitize: {dict(stats)}")
    return True


def convert_pdf_glmocr(pdf_path, out_dir, layout_device, settings):
    """TIER 1/2: GLM-OCR via CLI dell'SDK (venv dedicato o PATH).
    Ritorna False se il backend non e' disponibile -> il chiamante fa
    fallback sul percorso classico per quel PDF."""
    bin_path = (os.environ.get("GLMOCR_BIN") or settings.get("glmocr_bin")
                or shutil.which("glmocr"))
    if not bin_path:
        print("[glm-ocr] binario 'glmocr' non trovato -> fallback PyMuPDF")
        return False
    cfg = os.environ.get("GLMOCR_CONFIG") or settings.get("glmocr_config")
    if not cfg and os.path.exists(GLMOCR_CONFIG_PATH):
        cfg = GLMOCR_CONFIG_PATH
    tmp = tempfile.mkdtemp(prefix="glmocr_")
    cmd = [bin_path, "parse", pdf_path, "--output", tmp]
    if layout_device:
        cmd += ["--layout-device", layout_device]
    if cfg:
        cmd += ["--config", cfg]
    timeout = int(settings.get("glmocr_timeout", 1800))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-400:]
            print(f"[glm-ocr] rc={proc.returncode}: {tail} -> fallback PyMuPDF")
            return False
        md_file = None
        for root, _dirs, files in os.walk(tmp):
            for fname in sorted(files):
                if fname.lower().endswith(".md"):
                    md_file = os.path.join(root, fname)
                    break
            if md_file:
                break
        if not md_file:
            print("[glm-ocr] nessun .md prodotto -> fallback PyMuPDF")
            return False
        with open(md_file, encoding="utf-8") as f:
            text = f.read()
        text = _light_cleanup(strip_boilerplate(text))
        base_name = os.path.basename(pdf_path).replace(".pdf", ".md")
        out_path = os.path.join(out_dir, base_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Convertito (GLM-OCR, layout={layout_device}): {base_name}")
        return True
    except subprocess.TimeoutExpired:
        print(f"[glm-ocr] timeout {timeout}s -> fallback PyMuPDF")
        return False
    except Exception as e:
        print(f"[glm-ocr] eccezione: {e} -> fallback PyMuPDF")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def convert_pdf_to_md(pdf_path, out_dir, hw=None, settings=None):
    """Router hardware-aware. Mantiene la firma pubblica originaria.
    Priorita' modalita': env ARACHNE_OCR_MODE > settings.ocr_mode > auto."""
    settings = settings if settings is not None else load_settings()
    mode = (os.environ.get("ARACHNE_OCR_MODE")
            or settings.get("ocr_mode") or "auto").lower()
    if mode == "classic":
        return convert_pdf_classic(pdf_path, out_dir)
    hw = hw or (probe_hardware() if probe_hardware else {"tier": 3})
    layout = TIER_LAYOUT_DEVICE.get(hw.get("tier", 3)) or "cpu"
    if mode == "auto" and hw.get("tier", 3) == 3:
        return convert_pdf_classic(pdf_path, out_dir)
    # mode glm (forzato) oppure auto con tier 1/2: tenta GLM-OCR, poi fallback
    if not convert_pdf_glmocr(pdf_path, out_dir, layout, settings):
        return convert_pdf_classic(pdf_path, out_dir)
    return True


if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/raw_pdfs"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "../data/converted_md"
    if len(sys.argv) > 3:
        os.environ["ARACHNE_OCR_MODE"] = sys.argv[3]  # override CLI opzionale
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    hw = probe_hardware() if probe_hardware else {"tier": 3, "tier_label": "probe assente"}
    print(f"[hardware] {hw.get('tier_label')} | GPU: {hw.get('gpu_name')} | "
          f"VRAM: {hw.get('vram_used_mb')}/{hw.get('vram_total_mb')} MB | "
          f"glmocr: {hw.get('glmocr_available')}")

    pdfs = glob.glob(os.path.join(in_dir, "*.pdf"))
    if not pdfs:
        print(f"Nessun PDF trovato in {in_dir}")
    else:
        for pdf in pdfs:
            convert_pdf_to_md(pdf, out_dir, hw=hw)
