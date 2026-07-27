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

# Moduli condivisi con la pipeline (hardware_probe -> /api/system/hardware)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
try:
    from hardware_probe import probe_hardware
except Exception:
    probe_hardware = None

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


@app.get("/api/system/hardware")
def system_hardware():
    """Diagnostica hardware dinamica: GPU attiva, VRAM totale/usata/libera,
    tier informativo (1/2=OCR diretto via Ollama, 3=PyMuPDF+sanitize).
    Mai 5xx: in assenza di probe risponde comunque con un tier 3 sicuro."""
    force_cpu = False
    if os.path.exists(SETTINGS_FILE):
        try:
            force_cpu = json.load(open(SETTINGS_FILE)).get("force_cpu", False)
        except Exception:
            pass
    if probe_hardware is None:
        return {"gpu_present": False, "gpu_name": None, "tier": 3,
                "tier_label": "TIER 3 - PyMuPDF + sanitizzazione regex",
                "probe_source": "none", "ocr_available": False,
                "ollama_url": None, "ocr_model": None,
                "force_cpu": force_cpu,
                "error": "hardware_probe non importabile"}
    try:
        hw = probe_hardware()
    except Exception as e:
        return {"gpu_present": False, "gpu_name": None, "tier": 3,
                "tier_label": "TIER 3 - PyMuPDF + sanitizzazione regex",
                "probe_source": "error", "ocr_available": False,
                "ollama_url": None, "ocr_model": None,
                "force_cpu": force_cpu,
                "error": str(e)}
    hw["force_cpu"] = force_cpu
    return hw


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
    """Grafo attivo. Risponde SEMPRE 200 con payload sanitizzato:
    - file mancante/corrotto/senza nodi validi -> {'nodes': [], 'edges': []}
    - nodi: solo dict con id valido, deduplicati, label/metrics garantiti
    - archi: solo endpoint esistenti, niente self-loop, dedupe non orientato
    Sigma.js/Graphology non ricevera' mai una struttura che lo faccia crashare."""
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
        if not isinstance(data, dict):
            return JSONResponse(content=empty)
        nodes_raw = data.get("nodes")
        if not isinstance(nodes_raw, list):
            return JSONResponse(content=empty)
        edges_raw = data.get("edges")
        if not isinstance(edges_raw, list):
            links_raw = data.get("links")
            edges_raw = links_raw if isinstance(links_raw, list) else []
        # --- sanitizzazione nodi: id valido, dedupe, campi garantiti
        clean_nodes, seen_ids = [], set()
        for n in nodes_raw:
            if not isinstance(n, dict):
                continue
            nid = n.get("id")
            if nid is None or (isinstance(nid, str) and not nid.strip()) or nid in seen_ids:
                continue
            seen_ids.add(nid)
            n.setdefault("label", str(nid))
            if not isinstance(n.get("metrics"), dict):
                n["metrics"] = {}
            clean_nodes.append(n)
        if not clean_nodes:
            return JSONResponse(content=empty)
        # --- sanitizzazione archi: endpoint esistenti, no self-loop, dedupe
        clean_edges, seen_pairs = [], set()
        for e in edges_raw:
            if not isinstance(e, dict):
                continue
            s, t = e.get("source"), e.get("target")
            if s is None or t is None or s == t:
                continue
            if s not in seen_ids or t not in seen_ids:
                continue
            key = tuple(sorted((str(s), str(t))))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            e.setdefault("relation", "")
            clean_edges.append(e)
        out: dict = {"nodes": clean_nodes, "edges": clean_edges}
        if isinstance(data.get("meta"), dict):
            out["meta"] = data["meta"]
        out["active_run_id"] = _get_active_run()
        return JSONResponse(content=out)
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
    """ELIMINAZIONE ATOMICA di un progetto, in un'unica transazione logica:
    1. record SQLite rimosso dalla tabella runs;
    2. archivio grafo (run_{id}_graph.json) rimosso dal disco;
    3. se era il progetto attivo, stato live resettato (graph.json,
       graph_with_metrics.json, active_run.txt).
    Gestisce anche gli archivi orfani senza record DB."""
    conn = db()
    try:
        row = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row:
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()
    finally:
        conn.close()
    archive = os.path.join(RUNS_DIR, f"run_{run_id}_graph.json")
    had_archive = os.path.exists(archive)
    if had_archive:
        try:
            os.remove(archive)
        except OSError:
            pass
    was_active = _get_active_run() == run_id
    if was_active:
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
    return {"status": "deleted", "run_id": run_id, "was_active": was_active}


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


# ===========================================================================
# EXPORT HUB -- artefatti per-progetto (memoria analitica)
# ===========================================================================
WEB_CACHE_DIR = os.path.join(DATA_DIR, ".web_cache")
os.makedirs(WEB_CACHE_DIR, exist_ok=True)

VIEW_LIBS = {
    "graphology": "https://unpkg.com/graphology/dist/graphology.umd.min.js",
    "sigma": "https://unpkg.com/sigma/build/sigma.min.js",
    "forceatlas2": "https://unpkg.com/graphology-layout-forceatlas2/build/graphology-layout-forceatlas2.min.js",
}


def _load_run_graph(run_id):
    """Carica e valida il grafo archiviato di un progetto. Ritorna il dict
    JSON oppure None se assente/corrotto."""
    archive = os.path.join(RUNS_DIR, f"run_{run_id}_graph.json")
    if not _is_valid_graph_file(archive):
        return None
    try:
        with open(archive, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _nx_from_data(data):
    """Costruisce un DiGraph networkx dal JSON di un run (nodi+metriche+archi)."""
    import networkx as nx
    G = nx.DiGraph()
    for n in data.get("nodes", []):
        m = n.get("metrics", {}) or {}
        G.add_node(n["id"], label=str(n.get("label", n["id"])),
                   type=str(n.get("type", "concept")),
                   degree=float(m.get("degree", 0)),
                   betweenness=float(m.get("betweenness", 0)),
                   constraint=float(m.get("constraint", 1.0)),
                   community=int(m.get("community", 0)))
    for e in data.get("edges", data.get("links", [])):
        if e.get("source") in G and e.get("target") in G:
            G.add_edge(e["source"], e["target"],
                       relation=str(e.get("relation", "")),
                       weight=float(e.get("weight", 1)))
    return G


def _fetch_web_lib(name):
    """Scarica (con cache su disco) una libreria JS per il viewer offline.
    Ritorna il sorgente JS oppure None se non raggiungibile."""
    cache = os.path.join(WEB_CACHE_DIR, f"{name}.js")
    if os.path.exists(cache) and os.path.getsize(cache) > 1000:
        try:
            with open(cache, encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass
    try:
        import urllib.request
        req = urllib.request.Request(VIEW_LIBS[name],
                                     headers={"User-Agent": "arachne-scholar/4.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            src = r.read().decode("utf-8", errors="replace")
        if len(src) > 1000:
            with open(cache, "w", encoding="utf-8") as f:
                f.write(src)
            return src
    except Exception:
        pass
    return None


def _render_view_html(data, title):
    """Genera graph_view.html: file statico self-contained che impacchetta
    Graphology+Sigma+ForceAtlas2 (inlinati se scaricabili, altrimenti CDN)
    e il JSON del grafo, navigabile offline nel browser dell'utente."""
    libs_inline, libs_cdn = [], []
    for name, url in VIEW_LIBS.items():
        src = _fetch_web_lib(name)
        if src:
            libs_inline.append(f"/* ==== {name} (inlined) ==== */\n" + src)
        else:
            libs_cdn.append(f'<script src="{url}"></script>')
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    libs_block = ("\n".join(f"<script>\n{s}\n</script>" for s in libs_inline)
                  + "\n" + "\n".join(libs_cdn))
    return VIEW_TEMPLATE.replace("__TITLE__", title) \
                        .replace("__LIBS__", libs_block) \
                        .replace("__DATA__", payload)


VIEW_TEMPLATE = """<!DOCTYPE html>
<html lang="it"><head><meta charset="UTF-8">
<title>__TITLE__</title>
<style>
 body{margin:0;background:#0a0612;color:#e2e8f0;font-family:Inter,system-ui,sans-serif;}
 #g{position:fixed;inset:0;}
 #hud{position:fixed;top:0;left:0;right:0;padding:10px 16px;background:rgba(10,6,18,.85);
      backdrop-filter:blur(12px);border-bottom:1px solid rgba(168,85,247,.25);z-index:10;
      font:13px/1.4 monospace;color:#a855f7;}
 #hud b{color:#fff;}
 #info{position:fixed;top:52px;right:12px;width:300px;max-height:80vh;overflow-y:auto;display:none;
       background:rgba(14,8,26,.94);border:1px solid rgba(34,211,238,.45);border-radius:12px;
       padding:14px;z-index:20;font-size:12px;}
 #info h3{margin:0 0 4px;color:#fff;font-size:14px;word-break:break-word;}
 #info .rel{color:#bb86fc;font-family:monospace;}
 #info .row{padding:5px 4px;border-bottom:1px solid rgba(168,85,247,.12);cursor:pointer;}
 #info .row:hover{background:rgba(34,211,238,.12);}
 #info .passage{font-style:italic;color:#8b7bb8;border-left:2px solid #a855f7;padding-left:8px;margin:6px 0;}
 #x{float:right;cursor:pointer;color:#f472b6;font-family:monospace;}
</style></head><body>
<div id="hud"><b>__TITLE__</b> &mdash; ARACHNE // SCHOLAR offline viewer</div>
<div id="g"></div><div id="info"></div>
__LIBS__
<script>
const DATA = __DATA__;
const commColors={0:'#a855f7',1:'#22d3ee',2:'#f472b6',3:'#34d399',4:'#fbbf24',5:'#818cf8'};
const graph=new graphology.UndirectedGraph();
const deg={};
(DATA.edges||[]).forEach(e=>{deg[e.source]=(deg[e.source]||0)+1;deg[e.target]=(deg[e.target]||0)+1;});
(DATA.nodes||[]).forEach(n=>graph.mergeNode(n.id,{label:n.label||n.id,
  size:Math.min(28,Math.max(3,Math.sqrt(deg[n.id]||1)*2.2)),
  color:commColors[(n.metrics&&n.metrics.community)||0]||'#888',
  x:Math.random()*100,y:Math.random()*100}));
(DATA.edges||[]).forEach(e=>{if(e.source!==e.target&&graph.hasNode(e.source)&&graph.hasNode(e.target))
  try{graph.mergeEdge(e.source,e.target,{label:e.relation||'',size:1,color:'rgba(168,85,247,0.4)'});}catch(_){}});
if(window.graphologyForceAtlas2&&graphologyForceAtlas2.ForceAtlas2Layout){
  const lay=new graphologyForceAtlas2.ForceAtlas2Layout(graph,{settings:{gravity:.2,scalingRatio:10,slowDown:10}});
  lay.start(); setTimeout(()=>lay.stop(),9000);
}
const sigma=new Sigma(graph,document.getElementById('g'),{renderEdgeLabels:true,
  labelFont:'Inter',labelSize:12,edgeLabelFont:'monospace',edgeLabelSize:9,
  labelColor:{color:'#e9e4f5'},defaultEdgeColor:'rgba(168,85,247,0.35)'});
const labelOf=id=>graph.hasNode(id)?graph.getNodeAttribute(id,'label'):id;
sigma.on('clickNode',({node})=>{
  const info=document.getElementById('info');const rows=[];
  (DATA.edges||[]).forEach(e=>{
    if(e.source===node)rows.push({a:'&rarr;',e:e,other:e.target});
    else if(e.target===node)rows.push({a:'&larr;',e:e,other:e.source});});
  info.innerHTML='<span id="x" onclick="this.parentElement.style.display=\'none\'">[X]</span>'
    +'<h3>'+labelOf(node)+'</h3><div style="color:#22d3ee;font:10px monospace">'+rows.length+' connessioni</div>'
    +rows.slice(0,60).map(r=>'<div class="row"><span class="rel">'+r.a+' '+(r.e.relation||'')+'</span> '
      +labelOf(r.other)+(r.e.passage?'<div class="passage">&laquo;'+r.e.passage+'&raquo;</div>':'')+'</div>').join('');
  info.style.display='block';
});
</script></body></html>"""


def _project_name(run_id):
    conn = db()
    try:
        row = conn.execute("SELECT nome_progetto FROM runs WHERE id=?", (run_id,)).fetchone()
    finally:
        conn.close()
    if row and row[0]:
        return row[0]
    return f"progetto_{run_id}"


def _safe_slug(name):
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip("_")
    return slug or "arachne"


@app.get("/api/projects/{run_id}/export/json")
def export_project_json(run_id: int):
    """graph.json puro del run: memoria analitica grezza per agenti LLM esterni."""
    archive = os.path.join(RUNS_DIR, f"run_{run_id}_graph.json")
    if not _is_valid_graph_file(archive):
        return JSONResponse(status_code=404, content={"error": f"Archivio progetto #{run_id} assente o corrotto."})
    return FileResponse(archive, media_type="application/json",
                        filename=f"{_safe_slug(_project_name(run_id))}_graph.json")


@app.get("/api/projects/{run_id}/export/gexf")
def export_project_gexf(run_id: int):
    return _export_project_fmt(run_id, "gexf")


@app.get("/api/projects/{run_id}/export/graphml")
def export_project_graphml(run_id: int):
    return _export_project_fmt(run_id, "graphml")


def _export_project_fmt(run_id, fmt):
    """GEXF/GraphML puliti e purificati, pronti per l'importazione in Gephi."""
    data = _load_run_graph(run_id)
    if data is None:
        return JSONResponse(status_code=404, content={"error": f"Archivio progetto #{run_id} assente o corrotto."})
    G = _nx_from_data(data)
    suffix = f".{fmt}"
    tmp = os.path.join(tempfile.gettempdir(), f"arachne_run{run_id}{suffix}")
    import networkx as nx
    if fmt == "gexf":
        nx.write_gexf(G, tmp)
    else:
        nx.write_graphml(G, tmp)
    with open(tmp, "r", encoding="utf-8", errors="replace") as tf:
        raw = tf.read()
    with open(tmp, "w", encoding="utf-8") as tf:
        tf.write(purify_xml(raw))
    return FileResponse(tmp, media_type="application/xml",
                        filename=f"{_safe_slug(_project_name(run_id))}{suffix}")


@app.get("/api/projects/{run_id}/export/view")
def export_project_view(run_id: int):
    """graph_view.html: viewer Sigma.js statico self-contained, navigabile offline."""
    data = _load_run_graph(run_id)
    if data is None:
        return JSONResponse(status_code=404, content={"error": f"Archivio progetto #{run_id} assente o corrotto."})
    name = _project_name(run_id)
    html = _render_view_html(data, f"{name} — run #{run_id}")
    tmp = os.path.join(tempfile.gettempdir(), f"arachne_run{run_id}_view.html")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(html)
    return FileResponse(tmp, media_type="text/html",
                        filename=f"{_safe_slug(name)}_view.html")


@app.get("/api/projects/{run_id}/export/package")
def export_project_package(run_id: int):
    """Pacchetto completo: graph.json + .gexf + .graphml + graph_view.html (ZIP)."""
    import io
    import zipfile
    data = _load_run_graph(run_id)
    if data is None:
        return JSONResponse(status_code=404, content={"error": f"Archivio progetto #{run_id} assente o corrotto."})
    import networkx as nx
    name = _project_name(run_id)
    slug = _safe_slug(name)
    G = _nx_from_data(data)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{slug}_graph.json",
                   json.dumps(data, ensure_ascii=False, indent=2))
        for fmt, writer in (("gexf", nx.write_gexf), ("graphml", nx.write_graphml)):
            tmp = os.path.join(tempfile.gettempdir(), f"arachne_zip_{run_id}.{fmt}")
            writer(G, tmp)
            with open(tmp, "r", encoding="utf-8", errors="replace") as tf:
                z.writestr(f"{slug}.{fmt}", purify_xml(tf.read()))
        z.writestr(f"{slug}_view.html", _render_view_html(data, f"{name} — run #{run_id}"))
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{slug}_package.zip"'})
