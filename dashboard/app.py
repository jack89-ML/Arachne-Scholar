#!/usr/bin/env python3
"""
Arachne Scholar — Knowledge Graph Dashboard (FastAPI)
Backend API + serve del frontend statico.

Endpoints:
  GET /                      -> index.html
  GET /api/stats             -> metriche globali
  GET /api/types             -> tipi nodo disponibili
  GET /api/search?q=&type=   -> ricerca nodi per etichetta/tipo
  GET /api/node/{id}         -> dettaglio nodo + grado
  GET /api/neighborhood/{id} -> sottografo 1-hop con archi SVO
  GET /api/main_component    -> componente principale (hub centrali)
  GET /api/hubs              -> top nodi per grado

Avvio:  python3 app_dashboard.py   ->  http://0.0.0.0:8000
"""
import json, os, re
from collections import Counter, defaultdict

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

BASE_DIR = os.path.expanduser("~/scholar_engine")
GRAPH_PATH = os.path.join(BASE_DIR, "graph_out", "graph.json")
INDEX_HTML = os.path.join(BASE_DIR, "dashboard", "index.html")

# ----------------------------------------------------------------------------
# CARICAMENTO GRAFO IN MEMORIA
# ----------------------------------------------------------------------------
print(f"[dashboard] caricamento grafo da {GRAPH_PATH} ...")
with open(GRAPH_PATH, encoding="utf-8") as f:
    _g = json.load(f)

NODES = _g.get("nodes", [])
RAW_EDGES = _g.get("links", _g.get("edges", []))

# indicizza nodi
NODES_BY_ID = {}
for n in NODES:
    nid = n.get("id")
    if nid and nid not in NODES_BY_ID:
        NODES_BY_ID[nid] = n

# filtra archi con endpoint esistenti, costruisci adiacenza
ADJ = defaultdict(list)   # nid -> [(other, relation, weight, out_dir)]
EDGES = []
for e in RAW_EDGES:
    s, t = e.get("source"), e.get("target")
    if s in NODES_BY_ID and t in NODES_BY_ID and s != t:
        rel = e.get("relation", "related")
        w = e.get("weight", 1)
        EDGES.append((s, t, rel, w))
        ADJ[s].append((t, rel, w, True))
        ADJ[t].append((s, rel, w, False))

# grado per nodo
DEGREE = Counter()
for s, t, _, w in EDGES:
    DEGREE[s] += 1
    DEGREE[t] += 1

# componenti connesse (union-find) per la main component
_parent = {}
def find(x):
    _parent.setdefault(x, x)
    while _parent[x] != x:
        _parent[x] = _parent[_parent[x]]
        x = _parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        _parent[ra] = rb

for s, t, _, _ in EDGES:
    union(s, t)

_comp_sizes = Counter(find(n) for n in list(_parent))
MAIN_ROOT = _comp_sizes.most_common(1)[0][0] if _comp_sizes else None
MAIN_COMPONENT = {n for n in _parent if find(n) == MAIN_ROOT} if MAIN_ROOT else set()

TYPE_COUNTS = Counter(n.get("type", "concept") for n in NODES_BY_ID.values())
REL_COUNTS = Counter(e[2] for e in EDGES)

print(f"[dashboard] nodi={len(NODES_BY_ID)} archi={len(EDGES)} "
      f"main_component={len(MAIN_COMPONENT)}")

# ----------------------------------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------------------------------
app = FastAPI(title="Arachne Scholar KG Dashboard", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def node_view(nid, n=None):
    n = n or NODES_BY_ID[nid]
    return {
        "id": nid,
        "label": n.get("label", nid),
        "type": n.get("type", "concept"),
        "description": n.get("description", ""),
        "degree": DEGREE.get(nid, 0),
    }


@app.get("/")
def index():
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML)
    return JSONResponse({"error": "index.html non trovato",
                         "expected": INDEX_HTML}, status_code=404)


@app.get("/api/stats")
def stats():
    return {
        "nodes": len(NODES_BY_ID),
        "edges": len(EDGES),
        "main_component_size": len(MAIN_COMPONENT),
        "types": dict(TYPE_COUNTS.most_common()),
        "top_relations": dict(REL_COUNTS.most_common(12)),
    }


@app.get("/api/types")
def types():
    return {"types": sorted(TYPE_COUNTS.keys())}


@app.get("/api/search")
def search(q: str = Query("", description="testo da cercare"),
           type: str = Query("", description="filtro tipo"),
           limit: int = Query(40, le=200)):
    ql = q.strip().lower()
    tf = type.strip().lower()
    results = []
    for nid, n in NODES_BY_ID.items():
        if tf and n.get("type", "").lower() != tf:
            continue
        lbl = n.get("label", "").lower()
        if ql and ql not in lbl and ql not in nid.lower():
            continue
        results.append(node_view(nid, n))
    results.sort(key=lambda x: -x["degree"])
    return {"count": len(results), "results": results[:limit]}


@app.get("/api/node/{node_id}")
def node_detail(node_id: str):
    if node_id not in NODES_BY_ID:
        raise HTTPException(404, f"nodo '{node_id}' non trovato")
    nv = node_view(node_id)
    neighbors = []
    for other, rel, w, out_dir in ADJ.get(node_id, []):
        on = NODES_BY_ID[other]
        neighbors.append({
            "id": other,
            "label": on.get("label", other),
            "type": on.get("type", "concept"),
            "relation": rel,
            "weight": w,
            "direction": "out" if out_dir else "in",
            "degree": DEGREE.get(other, 0),
        })
    neighbors.sort(key=lambda x: -x["degree"])
    nv["neighbors"] = neighbors
    return nv


@app.get("/api/neighborhood/{node_id}")
def neighborhood(node_id: str, limit: int = Query(120, le=400)):
    if node_id not in NODES_BY_ID:
        raise HTTPException(404, f"nodo '{node_id}' non trovato")
    sub_nodes = {node_id: node_view(node_id)}
    sub_edges = []
    for other, rel, w, out_dir in ADJ.get(node_id, []):
        if other not in sub_nodes:
            sub_nodes[other] = node_view(other)
        sub_edges.append({
            "source": node_id if out_dir else other,
            "target": other if out_dir else node_id,
            "relation": rel,
            "weight": w,
        })
    nodes_list = sorted(sub_nodes.values(), key=lambda x: -x["degree"])[:limit]
    keep = {n["id"] for n in nodes_list}
    sub_edges = [e for e in sub_edges if e["source"] in keep and e["target"] in keep]
    return {"nodes": nodes_list, "edges": sub_edges}


@app.get("/api/main_component")
def main_component(limit: int = Query(250, le=800)):
    # hub della componente principale ordinati per grado
    ids = sorted(MAIN_COMPONENT, key=lambda x: -DEGREE.get(x, 0))[:limit]
    keep = set(ids)
    nodes_list = [node_view(i) for i in ids]
    edges_list = []
    for s, t, rel, w in EDGES:
        if s in keep and t in keep:
            edges_list.append({"source": s, "target": t,
                               "relation": rel, "weight": w})
    return {"nodes": nodes_list, "edges": edges_list}


@app.get("/api/hubs")
def hubs(limit: int = Query(60, le=300)):
    top = DEGREE.most_common(limit)
    return {"hubs": [node_view(nid) for nid, _ in top]}


if __name__ == "__main__":
    print("[dashboard] avvio su http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
