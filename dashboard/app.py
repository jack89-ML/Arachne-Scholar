from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import json
import sqlite3
import subprocess
import shutil
import tempfile
from datetime import datetime, timezone

app = FastAPI(title="Arachne Scholar Web Engine")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "raw_pdfs")
MD_DIR = os.path.join(DATA_DIR, "converted_md")
OUT_DIR = os.path.join(DATA_DIR, "graph_out")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
DB_PATH = os.path.join(DATA_DIR, "arachne.db")
LOG_FILE = os.path.join(BASE_DIR, "pipeline.log")
LANG_FILE = os.path.join(BASE_DIR, "pipeline.lang")

for d in [PDF_DIR, MD_DIR, OUT_DIR, RUNS_DIR]:
    os.makedirs(d, exist_ok=True)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------- persistenza
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS runs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               ts TEXT, lang TEXT, status TEXT,
               nodes INTEGER, edges INTEGER, graph_path TEXT)"""
    )
    return conn


def register_run(lang):
    conn = db()
    cur = conn.execute(
        "INSERT INTO runs (ts, lang, status) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), lang, "running"),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


def close_run(rid, graph_json=None, status="done"):
    nodes = edges = 0
    if graph_json and os.path.exists(graph_json):
        try:
            with open(graph_json, encoding="utf-8") as f:
                g = json.load(f)
            nodes, edges = len(g.get("nodes", [])), len(g.get("edges", []))
        except Exception:
            pass
    conn = db()
    conn.execute(
        "UPDATE runs SET status=?, nodes=?, edges=?, graph_path=? WHERE id=?",
        (status, nodes, edges, graph_json or "", rid),
    )
    conn.commit()
    conn.close()


def wipe_dir(path):
    """Svuota una directory senza rimuoverla (fix residui tra sessioni)."""
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.remove(p)
            except OSError:
                pass


# ------------------------------------------------------------------- rotte UI
@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/system-check")
def system_check():
    try:
        import spacy
        gpu = spacy.prefer_gpu()
    except Exception:
        gpu = False
    graph_exists = os.path.exists(os.path.join(OUT_DIR, "graph_with_metrics.json"))
    return {"gpu_available": gpu, "graph_exists": graph_exists}


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...), lang: str = Form("en")):
    # FIX PERSISTENZA: nessun residuo da sessioni precedenti. Prima di salvare
    # i nuovi PDF si svuotano input, markdown convertiti e vecchi output.
    wipe_dir(PDF_DIR)
    wipe_dir(MD_DIR)
    wipe_dir(OUT_DIR)
    os.makedirs(RUNS_DIR, exist_ok=True)

    with open(LANG_FILE, "w") as lf:
        lf.write(lang)
    saved = 0
    for file in files:
        if file.filename and file.filename.endswith(".pdf"):
            file_location = os.path.join(PDF_DIR, os.path.basename(file.filename))
            with open(file_location, "wb+") as file_object:
                shutil.copyfileobj(file.file, file_object)
            saved += 1
    return {"info": f"Workspace ripulito. Caricati {saved} file.", "saved": saved}


def run_pipeline():
    lang_code = "en"
    if os.path.exists(LANG_FILE):
        lang_code = open(LANG_FILE).read().strip() or "en"

    run_id = register_run(lang_code)
    final_graph = os.path.join(OUT_DIR, "graph_with_metrics.json")

    with open(LOG_FILE, "w") as log:
        log.write(f"Avvio Pipeline Arachne-Scholar (lang={lang_code}, run #{run_id})...\n")
        scripts = [
            ("Ingestione PDF", f"python {BASE_DIR}/src/ingest_pdf.py {PDF_DIR} {MD_DIR}"),
            ("Estrazione SVO", f"python {BASE_DIR}/src/extract_svo.py {MD_DIR} {OUT_DIR} {lang_code}"),
            ("Calcolo Metriche SNA", f"python {BASE_DIR}/src/sna_metrics.py {OUT_DIR}/graph.json {final_graph}"),
        ]
        try:
            for name, cmd in scripts:
                log.write(f"\n--- ESECUZIONE: {name} ---\n")
                log.flush()
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                for line in process.stdout or []:
                    log.write(line)
                    log.flush()
                process.wait()
                if process.returncode != 0:
                    log.write(f"\n!!! ERRORE nello step: {name} (rc={process.returncode})\n")
                    close_run(run_id, status="error")
                    return
            # archivio immutabile del grafo prodotto (anti-sovrascrittura)
            if os.path.exists(final_graph):
                archive = os.path.join(RUNS_DIR, f"run_{run_id}_graph.json")
                shutil.copy2(final_graph, archive)
            close_run(run_id, graph_json=final_graph, status="done")
            log.write("\n=== PIPELINE COMPLETATA ===")
        except Exception as e:
            log.write(f"\n!!! ECCEZIONE: {e}\n")
            close_run(run_id, status="error")


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


@app.get("/api/runs")
def list_runs():
    try:
        conn = db()
        rows = conn.execute(
            "SELECT id, ts, lang, status, nodes, edges FROM runs ORDER BY id DESC LIMIT 20"
        ).fetchall()
        conn.close()
        return {"runs": [
            {"id": r[0], "ts": r[1], "lang": r[2], "status": r[3], "nodes": r[4], "edges": r[5]}
            for r in rows
        ]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ------------------------------------------------------- export GEXF/GraphML
def _build_nx():
    import networkx as nx
    metrics_path = os.path.join(OUT_DIR, "graph_with_metrics.json")
    base_path = os.path.join(OUT_DIR, "graph.json")
    src = metrics_path if os.path.exists(metrics_path) else (
        base_path if os.path.exists(base_path) else None)
    if src is None:
        return None
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    G = nx.DiGraph()
    for n in data["nodes"]:
        m = n.get("metrics", {}) or {}
        G.add_node(n["id"], label=str(n.get("label", n["id"])),
                   type=str(n.get("type", "concept")),
                   degree=float(m.get("degree", 0)),
                   betweenness=float(m.get("betweenness", 0)),
                   constraint=float(m.get("constraint", 1.0)),
                   community=int(m.get("community", 0)))
    for e in data.get("edges", data.get("links", [])):
        G.add_edge(e["source"], e["target"],
                   relation=str(e.get("relation", "")),
                   weight=float(e.get("weight", 1)))
    return G


def _export(fmt):
    G = _build_nx()
    if G is None:
        return JSONResponse(status_code=404, content={"error": "Nessun grafo da esportare."})
    suffix = ".gexf" if fmt == "gexf" else ".graphml"
    tmp = os.path.join(tempfile.gettempdir(), f"arachne_export{suffix}")
    import networkx as nx
    if fmt == "gexf":
        nx.write_gexf(G, tmp)
    else:
        nx.write_graphml(G, tmp)
    return FileResponse(tmp, media_type="application/xml",
                        filename=f"arachne_scholar{suffix}")


@app.get("/api/export/gexf")
def export_gexf():
    return _export("gexf")


@app.get("/api/export/graphml")
def export_graphml():
    return _export("graphml")
