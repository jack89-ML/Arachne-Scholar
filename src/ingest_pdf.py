import os, sys, glob, re
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Errore: PyMuPDF non installato. Esegui 'pip install pymupdf'")
    sys.exit(1)


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


def convert_pdf_to_md(pdf_path, out_dir):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    # filtro boilerplate sul testo grezzo prima della scrittura
    text = strip_boilerplate(text)
    text = re.sub(r'\n{3,}', '\n\n', text)  # collassa whitespace eccessivo
    base_name = os.path.basename(pdf_path).replace(".pdf", ".md")
    out_path = os.path.join(out_dir, base_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Convertito (pulito): {base_name}")

if __name__ == "__main__":
    in_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/raw_pdfs"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "../data/converted_md"
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    pdfs = glob.glob(os.path.join(in_dir, "*.pdf"))
    if not pdfs:
        print(f"Nessun PDF trovato in {in_dir}")
    else:
        for pdf in pdfs:
            convert_pdf_to_md(pdf, out_dir)
