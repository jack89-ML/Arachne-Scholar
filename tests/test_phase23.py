#!/usr/bin/env python3
"""Arachne Scholar — regression tests Fase 2+3 (robustezza & igiene).

Copre:
  Fase 2  - /api/logs incrementale (offset, delta, reset su nuovo run)
          - export su tmp univoci con cleanup (niente residui in /tmp)
          - viewer offline: librerie VENDORED lette senza rete
          - viewer offline: blindatura XSS (titolo, label, relation, passage)
          - sna_metrics: fallback "links" per grafi senza chiave "edges"
  Fase 3  - i file vendored esistono davvero nel repo (no CDN nel percorso base)

Esecuzione:  python3 -m pytest tests/ -v
"""
import glob
import io
import json
import os
import sys
import tempfile

import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "dashboard"))

import app as A
from fastapi.testclient import TestClient


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    data = tmp_path / "data"
    pdf, md = data / "raw_pdfs", data / "converted_md"
    out, runs = data / "graph_out", data / "graph_out" / "runs"
    for d in (pdf, md, out, runs):
        d.mkdir(parents=True)
    paths = {
        "BASE_DIR": str(tmp_path),
        "DATA_DIR": str(data),
        "PDF_DIR": str(pdf),
        "MD_DIR": str(md),
        "OUT_DIR": str(out),
        "RUNS_DIR": str(runs),
        "DB_PATH": str(data / "arachne.db"),
        "LOG_FILE": str(tmp_path / "pipeline.log"),
        "LANG_FILE": str(tmp_path / "pipeline.lang"),
        "SETTINGS_FILE": str(tmp_path / "settings.json"),
        "ACTIVE_FILE": str(out / "active_run.txt"),
        "WEB_CACHE_DIR": str(data / ".web_cache"),
    }
    for name, val in paths.items():
        monkeypatch.setattr(A, name, val)
    if A.PIPELINE_LOCK.locked():
        A.PIPELINE_LOCK.release()
    A.RUN_STATE["run_id"] = None
    return tmp_path


@pytest.fixture()
def client(ws):
    return TestClient(A.app)


# ------------------------------------------------------------- log incrementale
class TestIncrementalLogs:
    def test_offset_delta_and_reset(self, client, ws):
        log = ws / "pipeline.log"
        log.write_text("riga1\nriga2\n")
        r = client.get("/api/logs").json()
        assert r["logs"] == "riga1\nriga2\n"
        end = r["offset"]
        # nessun delta: offset invariato, logs vuoto
        r = client.get(f"/api/logs?offset={end}").json()
        assert r["logs"] == "" and r["offset"] == end and r["reset"] is False
        # delta: solo la coda nuova
        with open(log, "a") as f:
            f.write("riga3\n")
        r = client.get(f"/api/logs?offset={end}").json()
        assert r["logs"] == "riga3\n"
        # file troncato (nuovo run): offset > size -> reset da zero
        log.write_text("NUOVO RUN\n")
        r = client.get(f"/api/logs?offset={end + 100}").json()
        assert r["reset"] is True and r["logs"] == "NUOVO RUN\n"

    def test_missing_log_file(self, client):
        r = client.get("/api/logs").json()
        assert "In attesa" in r["logs"] and r["offset"] == 0


# ------------------------------------------------------------- export tmp uuid
def _make_run_helper(rid):
    """Registra un run done con archivio valido nel workspace della fixture."""
    graph = {
        "nodes": [{"id": "a", "label": "Alpha", "type": "concept",
                   "metrics": {"degree": 1, "betweenness": 0.5,
                               "constraint": 1.0, "community": 0}},
                  {"id": "b", "label": "Beta", "type": "concept",
                   "metrics": {"degree": 1, "betweenness": 0.0,
                               "constraint": 1.0, "community": 1}}],
        "edges": [{"source": "a", "target": "b", "relation": "analyze",
                   "weight": 2}],
    }
    archive = os.path.join(A.RUNS_DIR, f"run_{rid}_graph.json")
    with open(archive, "w") as f:
        json.dump(graph, f)
    conn = A.db()
    conn.execute(
        "INSERT INTO runs (ts, lang, status, nodes, edges, graph_path) "
        "VALUES ('2026-07-28T00:00:00+00:00', 'en', 'done', 2, 1, ?)",
        (archive,))
    conn.commit()
    conn.close()


class TestExportTempFiles:
    @pytest.mark.parametrize("fmt", ["gexf", "graphml"])
    def test_export_succeeds_and_tmp_cleaned(self, client, fmt):
        _make_run_helper(1)
        r = client.get(f"/api/projects/1/export/{fmt}")
        assert r.status_code == 200
        assert len(r.content) > 100
        leftover = glob.glob(os.path.join(tempfile.gettempdir(),
                                          "arachne_run1_*"))
        assert leftover == [], f"tmp non pulito: {leftover}"

    def test_export_package_zip_in_memory(self, client):
        _make_run_helper(1)
        r = client.get("/api/projects/1/export/package")
        assert r.status_code == 200
        import zipfile
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = z.namelist()
        assert any(n.endswith("_graph.json") for n in names)
        assert any(n.endswith(".gexf") for n in names)
        assert any(n.endswith(".graphml") for n in names)
        assert any(n.endswith("_view.html") for n in names)
        leftover = glob.glob(os.path.join(tempfile.gettempdir(), "arachne_zip_*"))
        assert leftover == []


# ------------------------------------------------------------- viewer offline
class TestVendoredLibs:
    def test_vendor_files_exist(self):
        for name, (fname, _url) in A.VIEW_LIBS.items():
            p = os.path.join(A.VENDOR_DIR, fname)
            assert os.path.exists(p), f"{fname} non vendored"
            assert os.path.getsize(p) > 1000

    def test_fetch_prefers_local_no_network(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("la rete NON deve essere consultata")
        monkeypatch.setattr("urllib.request.urlopen", boom)
        src = A._fetch_web_lib("graphology")
        assert src is not None and len(src) > 1000


class TestViewerXSS:
    EVIL_LABEL = '<img src=x onerror="alert(1)">'
    EVIL_PASSAGE = 'studio</script><script>alert(2)</script>'
    EVIL_TITLE = 'proj</title><script>alert(3)</script>'

    def _render(self):
        data = {
            "nodes": [{"id": "n1", "label": self.EVIL_LABEL, "type": "concept",
                       "metrics": {"community": 0}}],
            "edges": [{"source": "n1", "target": "n1", "relation": "<b>x</b>",
                       "passage": self.EVIL_PASSAGE}],
        }
        return A._render_view_html(data, self.EVIL_TITLE)

    def test_title_escaped(self):
        out = self._render()
        assert "</title><script>" not in out
        assert "&lt;/title&gt;" in out

    def test_script_breakout_neutralized(self):
        out = self._render()
        # il JSON inline non puo' chiudere il tag <script> del payload
        assert "alert(2)</script>" not in out
        assert "alert(2)<\\/script>" in out

    def test_every_dom_interpolation_is_escaped(self):
        # nel template, nessuna variabile del grafo finisce in innerHTML cruda
        tpl = A.VIEW_TEMPLATE
        assert "'+labelOf(" not in tpl
        assert "+(r.e.relation" not in tpl
        assert "+r.e.passage+" not in tpl
        assert "esc(labelOf(" in tpl
        assert "esc(r.e.relation" in tpl
        assert "esc(r.e.passage)" in tpl
        # la funzione esc copre i 5 caratteri critici
        for ch in ["&amp;", "&lt;", "&gt;", "&quot;", "&#39;"]:
            assert ch in tpl


# ------------------------------------------------------------- sna_metrics
import sna_metrics as M


class TestSnaFallback:
    def test_links_only_graph(self, tmp_path):
        src = tmp_path / "graph.json"
        src.write_text(json.dumps({
            "nodes": [{"id": "x", "type": "concept", "label": "X"},
                      {"id": "y", "type": "concept", "label": "Y"}],
            "links": [{"source": "x", "target": "y",
                       "relation": "analyze", "weight": 1}],
        }))
        out = tmp_path / "out.json"
        M.calculate_metrics(str(src), str(out))
        data = json.loads(out.read_text())
        assert all("metrics" in n for n in data["nodes"])
        assert data["meta"]["sna_mode"].startswith("top")
