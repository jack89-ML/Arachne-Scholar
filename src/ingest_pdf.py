import os, sys, glob
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Errore: PyMuPDF non installato. Esegui 'pip install pymupdf'")
    sys.exit(1)

def convert_pdf_to_md(pdf_path, out_dir):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    
    base_name = os.path.basename(pdf_path).replace(".pdf", ".md")
    out_path = os.path.join(out_dir, base_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Convertito: {base_name}")

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
