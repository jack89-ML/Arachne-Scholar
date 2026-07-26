from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
import re
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
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
ACTIVE_FILE = os.path.join(OUT_DIR, "active_run.txt")

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
    # Migrazione schema: colonna nome_progetto (rinominabile da UI)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(runs)").fetchall()]
    if "nome_progetto" not in cols:
        conn.execute("ALTER TABLE runs ADD COLUMN nome_progetto TEXT DEFAULT ''")
        conn.commit()
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


def wipe_dir(path, ignore=frozenset()):
    """Svuota una directory senza rimuoverla. `ignore` contiene nomi da preservare
    (es. 'runs': lo storico dei grafi NON deve mai essere cancellato)."""
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        if name in ignore:
            continue
        p = os.path.join(path, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.remove(p)
            except OSError:
                pass


def _is_valid_graph_file(path):
    """True se il file esiste, e' JSON valido e ha forma di grafo
    (dict con lista 'nodes' e lista 'edges' oppure 'links')."""
    try:
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
        if not isinstance(g, dict) or not isinstance(g.get("nodes"), list):
            return False
        return isinstance(g.get("edges"), list) or isinstance(g.get("links"), list)
    except Exception:
        return False


def _set_active_run(run_id):
    """Marca quale run e' attualmente servito come grafo live."""
    try:
        with open(ACTIVE_FILE, "w") as f:
            f.write(str(run_id))
    except OSError:
        pass


def _get_active_run():
    try:
        with open(ACTIVE_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


# ------------------------------------------------------------------- rotte UI
@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


def detect_gpu():
    """Rilevamento GPU robusto a cascata: torch -> cupy -> spacy.
    Ritorna (disponibile: bool, nome_scheda: str|None)."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            try:
                props = cupy.cuda.runtime.getDeviceProperties(0)
                name = props.get("name", b"")
                name = name.decode() if isinstance(name, bytes) else str(name)
            except Exception:
                name = "NVIDIA GPU"
            return True, name or "NVIDIA GPU"
    except Exception:
        pass
    try:
        import spacy
        if spacy.prefer_gpu():
            return True, "NVIDIA GPU"
    except Exception:
        pass
    return False, None


@app.get("/api/system-check")
def system_check():
    gpu, gpu_name = detect_gpu()
    graph_exists = os.path.exists(os.path.join(OUT_DIR, "graph_with_metrics.json"))
    force_cpu = False
    if os.path.exists(SETTINGS_FILE):
        try:
            force_cpu = json.load(open(SETTINGS_FILE)).get("force_cpu", False)
        except Exception:
            pass
    return {"gpu_available": gpu, "gpu_name": gpu_name, "graph_exists": graph_exists,
            "force_cpu": force_cpu}


@app.post("/api/upload")
async def upload_files(files: list[UploadFile] = File(...), lang: str = Form("en")):
    # FIX PERSISTENZA: nessun residuo da sessioni precedenti. Prima di salvare
    # i nuovi PDF si svuotano input e markdown convertiti. graph_out viene
    # pulito MA la sottocartella runs/ (storico progetti) e' preservata.
    wipe_dir(PDF_DIR)
    wipe_dir(MD_DIR)
    wipe_dir(OUT_DIR, ignore=frozenset({"runs"}))
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

    # Settings: forza CPU se richiesto (CUDA invisibile ai subprocess NLP)
    settings = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            settings = json.load(open(SETTINGS_FILE))
        except Exception:
            pass
    child_env = os.environ.copy()
    if settings.get("force_cpu"):
        child_env["CUDA_VISIBLE_DEVICES"] = ""

    run_id = register_run(lang_code)
    final_graph = os.path.join(OUT_DIR, "graph_with_metrics.json")

    with open(LOG_FILE, "w") as log:
        log.write(f"Avvio Pipeline Arachne-Scholar (lang={lang_code}, run #{run_id})...\n")
        scripts = [
            ("Ingestione PDF", f"{sys.executable} {BASE_DIR}/src/ingest_pdf.py {PDF_DIR} {MD_DIR}"),
            ("Estrazione SVO", f"{sys.executable} {BASE_DIR}/src/extract_svo.py {MD_DIR} {OUT_DIR} {lang_code}"),
            ("Calcolo Metriche SNA", f"{sys.executable} {BASE_DIR}/src/sna_metrics.py {OUT_DIR}/graph.json {final_graph}"),
        ]
        try:
            for name, cmd in scripts:
                log.write(f"\n--- ESECUZIONE: {name} ---\n")
                log.flush()
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, env=child_env)
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
            _set_active_run(run_id)
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
    """Grafo attivo. Risponde SEMPRE 200 con JSON formalmente valido:
    se il file manca, e' corrotto o non ha la forma attesa, restituisce
    un grafo vuoto {'nodes': [], 'edges': []} cosi' Sigma.js non crasha."""
    empty = {"nodes": [], "edges": []}
    metrics_path = os.path.join(OUT_DIR, "graph_with_metrics.json")
    base_path = os.path.join(OUT_DIR, "graph.json")
    src = metrics_path if os.path.exists(metrics_path) else (
        base_path if os.path.exists(base_path) else None)
    if src is None:
        return JSONResponse(content=empty)
    try:
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
            return JSONResponse(content=empty)
        if not isinstance(data.get("edges"), list):
            links = data.get("links")
            data["edges"] = links if isinstance(links, list) else []
        return JSONResponse(content=data)
    except Exception:
        return JSONResponse(content=empty)


def _backfill_legacy_run():
    """Se il registry e' vuoto ma esiste gia' un grafo sul disco, lo registra
    come run 'legacy' cosi' la griglia progetti della Home lo mostra subito."""
    metrics_path = os.path.join(OUT_DIR, "graph_with_metrics.json")
    conn = db()
    n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    if n == 0 and os.path.exists(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as f:
                g = json.load(f)
            ts = datetime.fromtimestamp(os.path.getmtime(metrics_path), tz=timezone.utc).isoformat()
            cur = conn.execute(
                "INSERT INTO runs (ts, lang, status, nodes, edges, graph_path) VALUES (?, ?, ?, ?, ?, ?)",
                (ts, g.get("meta", {}).get("lang", "en"), "done",
                 len(g.get("nodes", [])), len(g.get("edges", [])), metrics_path),
            )
            rid = cur.lastrowid
            shutil.copy2(metrics_path, os.path.join(RUNS_DIR, f"run_{rid}_graph.json"))
            conn.commit()
            _set_active_run(rid)
        except Exception:
            pass
    conn.close()


@app.get("/api/runs")
def list_runs():
    try:
        _backfill_legacy_run()
        conn = db()
        rows = conn.execute(
            "SELECT id, ts, lang, status, nodes, edges, nome_progetto FROM runs ORDER BY id DESC LIMIT 20"
        ).fetchall()
        # SELF-HEALING: i run terminali il cui archivio e' sparito dal disco
        # (cancellazione manuale, restore parziale, vecchi bug) non sono
        # apribili -> espunti dal registro. Mai piu' card fantasma.
        ghosts = [r[0] for r in rows
                  if r[3] in ("done", "error")
                  and not os.path.exists(os.path.join(RUNS_DIR, f"run_{r[0]}_graph.json"))]
        if ghosts:
            conn.executemany("DELETE FROM runs WHERE id=?", [(g,) for g in ghosts])
            conn.commit()
            rows = [r for r in rows if r[0] not in ghosts]
        conn.close()
        return {"runs": [
            {"id": r[0], "ts": r[1], "lang": r[2], "status": r[3], "nodes": r[4], "edges": r[5],
             "nome_progetto": r[6] or f"Progetto #{r[0]}",
             "has_graph": os.path.exists(os.path.join(RUNS_DIR, f"run_{r[0]}_graph.json"))}
            for r in rows
        ]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/runs/{run_id}/rename")
async def rename_run(run_id: int, payload: dict):
    """Rinomina un progetto al volo (tasto matita nelle card Home)."""
    name = str(payload.get("name", "")).strip()[:80]
    if not name:
        return JSONResponse(status_code=400, content={"error": "Nome vuoto."})
    conn = db()
    conn.execute("UPDATE runs SET nome_progetto=? WHERE id=?", (name, run_id))
    conn.commit()
    conn.close()
    return {"status": "renamed", "run_id": run_id, "nome_progetto": name}


@app.post("/api/runs/{run_id}/activate")
def activate_run(run_id: int):
    """Promuove un run archiviato a grafo attivo (aperto dalla griglia progetti).
    Robusto: verifica il record DB, ricostruisce l'archivio dal graph_path
    registrato se manca, e valida il JSON prima di copiarlo sui file live."""
    conn = db()
    row = conn.execute("SELECT graph_path FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    if not row:
        return JSONResponse(status_code=404, content={"error": f"Progetto #{run_id} non trovato nel registro."})
    archive = os.path.join(RUNS_DIR, f"run_{run_id}_graph.json")
    if not _is_valid_graph_file(archive):
        # Fallback: ricostruisci l'archivio dal percorso grafo registrato nel DB
        gp = row[0] or ""
        if gp and os.path.abspath(gp) != os.path.abspath(archive) and _is_valid_graph_file(gp):
            os.makedirs(RUNS_DIR, exist_ok=True)
            shutil.copy2(gp, archive)
        if not _is_valid_graph_file(archive):
            return JSONResponse(status_code=404, content={"error": f"Archivio del progetto #{run_id} assente o corrotto."})
    shutil.copy2(archive, os.path.join(OUT_DIR, "graph_with_metrics.json"))
    shutil.copy2(archive, os.path.join(OUT_DIR, "graph.json"))
    _set_active_run(run_id)
    return {"status": "activated", "run_id": run_id}


@app.delete("/api/projects/{run_id}")
async def delete_project(run_id: int):
    """Elimina un progetto: record DB + archivio grafo. I file live vengono
    rimossi SOLO se il progetto eliminato e' quello attivo (tracciato via
    active_run.txt): cancellare un progetto non aperto non svuota piu' la
    dashboard. Gestisce anche gli archivi orfani senza record DB."""
    conn = db()
    row = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.commit()
    conn.close()
    archive = os.path.join(RUNS_DIR, f"run_{run_id}_graph.json")
    had_archive = os.path.exists(archive)
    if had_archive:
        os.remove(archive)
    if _get_active_run() == run_id:
        for fname in ["graph.json", "graph_with_metrics.json"]:
            p = os.path.join(OUT_DIR, fname)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        try:
            os.remove(ACTIVE_FILE)
        except OSError:
            pass
    if not row and not had_archive:
        return JSONResponse(status_code=404, content={"error": f"Progetto #{run_id} non trovato."})
    return {"status": "deleted", "run_id": run_id}


# ------------------------------------------------------------- settings API
@app.get("/api/settings")
def get_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            return json.load(open(SETTINGS_FILE))
        except Exception:
            pass
    return {"nlp_model": "auto", "force_cpu": False}


@app.post("/api/settings")
async def save_settings(payload: dict):
    data = {
        "nlp_model": str(payload.get("nlp_model", "auto"))[:40],
        "force_cpu": bool(payload.get("force_cpu", False)),
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f)
    return {"status": "saved", **data}


def purify_xml(text):
    """(FIX) Purifica una stringa da caratteri di controllo ASCII illegali
    che mandano in crash i parser XML rigorosi (Gephi, GraphML). Rimuove
    l'intervallo [\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f] lasciando stampabili,
    \\n, \\t, \\r. Applicabile a XML, GraphML, JSON prima della scrittura I/O."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)


# ------------------------------------------------------- export GEXF/GraphML
def _build_nx(scope="all"):
    import networkx as nx
    metrics_path = os.path.join(OUT_DIR, "graph_with_metrics.json")
    base_path = os.path.join(OUT_DIR, "graph.json")
    src = metrics_path if os.path.exists(metrics_path) else (
        base_path if os.path.exists(base_path) else None)
    if src is None:
        return None
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    # scope="hub": stesso filtro del canvas (grado>0, top 400 per betweenness)
    nodes = data["nodes"]
    if scope == "hub":
        hubs = [n for n in nodes if (n.get("metrics", {}) or {}).get("degree", 0) > 0]
        hubs.sort(key=lambda n: -((n.get("metrics", {}) or {}).get("betweenness", 0)))
        keep = {n["id"] for n in hubs[:400]}
        nodes = [n for n in nodes if n["id"] in keep]
    else:
        keep = None

    G = nx.DiGraph()
    for n in nodes:
        m = n.get("metrics", {}) or {}
        G.add_node(n["id"], label=str(n.get("label", n["id"])),
                   type=str(n.get("type", "concept")),
                   degree=float(m.get("degree", 0)),
                   betweenness=float(m.get("betweenness", 0)),
                   constraint=float(m.get("constraint", 1.0)),
                   community=int(m.get("community", 0)))
    for e in data.get("edges", data.get("links", [])):
        if keep is not None and (e["source"] not in keep or e["target"] not in keep):
            continue
        G.add_edge(e["source"], e["target"],
                   relation=str(e.get("relation", "")),
                   weight=float(e.get("weight", 1)))
    return G


def _export(fmt, scope="all"):
    G = _build_nx(scope)
    if G is None:
        return JSONResponse(status_code=404, content={"error": "Nessun grafo da esportare."})
    suffix = ".gexf" if fmt == "gexf" else ".graphml"
    tmp = os.path.join(tempfile.gettempdir(), f"arachne_export_{scope}{suffix}")
    import networkx as nx
    if fmt == "gexf":
        nx.write_gexf(G, tmp)
    else:
        nx.write_graphml(G, tmp)
    # (FIX hard-ASCII) Purificazione dell'intero buffer XML prima di servirlo
    with open(tmp, "r", encoding="utf-8", errors="replace") as tf:
        raw = tf.read()
    with open(tmp, "w", encoding="utf-8") as tf:
        tf.write(purify_xml(raw))
    return FileResponse(tmp, media_type="application/xml",
                        filename=f"arachne_scholar_{scope}{suffix}")


@app.get("/api/export/gexf")
def export_gexf(scope: str = "all"):
    return _export("gexf", scope)


@app.get("/api/export/graphml")
def export_graphml(scope: str = "all"):
    return _export("graphml", scope)
