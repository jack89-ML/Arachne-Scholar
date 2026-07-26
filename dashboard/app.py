from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import subprocess
import shutil

app = FastAPI(title="Arachne Scholar Web Engine")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
MD_DIR = os.path.join(DATA_DIR, "converted_md")
OUT_DIR = os.path.join(DATA_DIR, "graph_out")
LOG_FILE = os.path.join(BASE_DIR, "pipeline.log")
LANG_FILE = os.path.join(BASE_DIR, "pipeline.lang")

for d in [PDF_DIR, MD_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/system-check")
def system_check():
    # Rilevamento GPU fittizio/reale tramite spacy
    try:
        import spacy
        gpu = spacy.prefer_gpu()
    except:
        gpu = False
    
    graph_exists = os.path.exists(os.path.join(OUT_DIR, "graph_with_metrics.json"))
    return {"gpu_available": gpu, "graph_exists": graph_exists}

@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...), lang: str = Form("en")):
    with open(LANG_FILE, "w") as lf:
        lf.write(lang)
    for file in files:
        if file.filename.endswith(".pdf"):
            file_location = f"{PDF_DIR}/{file.filename}"
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(file.file, file_object)
    return {"info": f"Caricati {len(files)} file."}

def run_pipeline():
    lang_code = "en"
    if os.path.exists(LANG_FILE):
        lang_code = open(LANG_FILE).read().strip() or "en"
    with open(LOG_FILE, "w") as log:
        log.write(f"Avvio Pipeline Arachne-Scholar (lang={lang_code})...\n")
        
        scripts = [
            ("Ingestione PDF", f"python {BASE_DIR}/src/ingest_pdf.py {PDF_DIR} {MD_DIR}"),
            ("Estrazione SVO", f"python {BASE_DIR}/src/extract_svo.py {MD_DIR} {OUT_DIR} {lang_code}"),
            ("Calcolo Metriche SNA", f"python {BASE_DIR}/src/sna_metrics.py {OUT_DIR}/graph.json {OUT_DIR}/graph_with_metrics.json")
        ]
        
        for name, cmd in scripts:
            log.write(f"\n--- ESECUZIONE: {name} ---\n")
            log.flush()
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
            process.wait()
        
        log.write("\n=== PIPELINE COMPLETATA ===")

@app.post("/api/run")
async def start_pipeline(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_pipeline)
    return {"status": "started"}

@app.get("/api/logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return {"logs": "In attesa di avvio..."}
    with open(LOG_FILE, "r") as f:
        return {"logs": f.read()}

@app.get("/api/graph")
def get_graph():
    metrics_path = os.path.join(OUT_DIR, "graph_with_metrics.json")
    base_path = os.path.join(OUT_DIR, "graph.json")
    if os.path.exists(metrics_path):
        return FileResponse(metrics_path)
    elif os.path.exists(base_path):
        return FileResponse(base_path)
    return JSONResponse(status_code=404, content={"error": "Grafo non trovato."})
