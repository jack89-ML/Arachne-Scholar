#!/usr/bin/env python3
"""Arachne Scholar — regression tests Fase 0+1 (security & core bugfix).

Copre:
  Fase 0  - whitelist lang in /api/upload (niente injection verso i subprocess)
          - run_pipeline: argv list (mai shell=True) + lang sanificata
          - lock pipeline: /api/run -> 409 se occupata
  Fase 1  - upload: valida PDF PRIMA del wipe; .PDF maiuscoli accettati
          - /api/graph: archi bidirezionali e multi-relazione preservati
          - /api/runs: i run "error" NON vengono ghostati
          - delete_project: 409 sul run corrente
          - normalize(): folding accenti (i18n IT/ES)
          - ingest_pdf: naming .md case-insensitive

Esecuzione:  python3 -m pytest tests/ -v
Niente spaCy, niente Ollama, niente Docker: extract_svo e' importabile
da quando il load del modello vive in main().
"""
import io
import json
import os
import sys

import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.join(BASE_DIR, "dashboard"))

import app as A  # dashboard/app.py
from fastapi.testclient import TestClient


# ---------------------------------------------------------------- fixtures
@pytest.fixture()
def ws(tmp_path, monkeypatch):
    """Workspace isolato: tutti i path di modulo puntano a tmp_path."""
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
    }
    for name, val in paths.items():
        monkeypatch.setattr(A, name, val)
    # sicurezza: mai una pipeline "vera" residua da altri test
    if A.PIPELINE_LOCK.locked():
        A.PIPELINE_LOCK.release()
    A.RUN_STATE["run_id"] = None
    return tmp_path


@pytest.fixture()
def client(ws):
    return TestClient(A.app)


# ------------------------------------------------------------- Fase 0 tests
class TestLangWhitelist:
    def test_malicious_lang_rejected_and_no_wipe(self, client, ws):
        marker = ws / "data" / "raw_pdfs" / "sentinella.pdf"
        marker.write_bytes(b"%PDF-1.4 fake")
        r = client.post(
            "/api/upload",
            files=[("files", ("paper.pdf", io.BytesIO(b"%PDF-1.4 x"), "application/pdf"))],
            data={"lang": "en; touch /tmp/pwned #"},
        )
        assert r.status_code == 400
        # niente wipe: la sentinella esiste ancora, pipeline.lang mai scritto
        assert marker.exists()
        assert not (ws / "pipeline.lang").exists()

    def test_lang_case_normalized(self, client):
        r = client.post(
            "/api/upload",
            files=[("files", ("paper.pdf", io.BytesIO(b"%PDF-1.4 x"), "application/pdf"))],
            data={"lang": " IT "},
        )
        assert r.status_code == 200
        assert open(A.LANG_FILE).read() == "it"

    def test_unsupported_lang_rejected(self, client):
        r = client.post(
            "/api/upload",
            files=[("files", ("paper.pdf", io.BytesIO(b"%PDF-1.4 x"), "application/pdf"))],
            data={"lang": "fr"},
        )
        assert r.status_code == 400


class TestUploadValidation:
    def test_no_valid_pdf_means_no_wipe(self, client, ws):
        marker = ws / "data" / "graph_out" / "graph.json"
        marker.write_text("{}")
        r = client.post(
            "/api/upload",
            files=[("files", ("notes.txt", io.BytesIO(b"hello"), "text/plain"))],
            data={"lang": "en"},
        )
        assert r.status_code == 400
        assert marker.exists()  # workspace intatto

    def test_uppercase_pdf_accepted(self, client, ws):
        r = client.post(
            "/api/upload",
            files=[("files", ("PAPER.PDF", io.BytesIO(b"%PDF-1.4 x"), "application/pdf"))],
            data={"lang": "en"},
        )
        assert r.status_code == 200
        assert r.json()["saved"] == 1
        assert (ws / "data" / "raw_pdfs" / "PAPER.PDF").exists()


class FakeProc:
    """Sostituto di subprocess.Popen: registra argv/env, finge rc=0."""
    calls = []

    def __init__(self, cmd, **kwargs):
        FakeProc.calls.append({"cmd": cmd, "kwargs": kwargs})
        self.returncode = 0
        self.stdout = iter([])

    def wait(self):
        return 0


class TestRunPipelineHardening:
    def test_argv_lists_no_shell_lang_sanitized(self, ws, monkeypatch):
        # pipeline.lang avvelenato a mano: la difesa in profondita' deve
        # ricadere su "en" senza mai vedere la stringa malevola negli argv.
        with open(A.LANG_FILE, "w") as f:
            f.write("en; rm -rf ~ #")
        with open(A.SETTINGS_FILE, "w") as f:
            json.dump({"nlp_model": "trf", "force_cpu": True}, f)
        FakeProc.calls = []
        monkeypatch.setattr(A.subprocess, "Popen", FakeProc)
        A.run_pipeline()
        assert len(FakeProc.calls) == 3  # ingest, svo, sna
        for call in FakeProc.calls:
            assert isinstance(call["cmd"], list), "argv deve essere una lista"
            assert "shell" not in call["kwargs"] or call["kwargs"]["shell"] is not True
            assert not any("rm -rf" in str(part) for part in call["cmd"])
        svo_cmd = FakeProc.calls[1]["cmd"]
        assert svo_cmd[-1] == "en"  # fallback whitelist
        env = FakeProc.calls[0]["kwargs"]["env"]
        assert env["ARACHNE_NLP_MODEL"] == "trf"          # cablatura settings->subprocess
        assert env["CUDA_VISIBLE_DEVICES"] == ""           # force_cpu rispettato
        # il lock viene sempre rilasciato, anche a fine run
        assert not A.PIPELINE_LOCK.locked()
        assert A.RUN_STATE["run_id"] is None

    def test_run_endpoint_conflict_when_locked(self, client):
        A.PIPELINE_LOCK.acquire()
        try:
            r = client.post("/api/run")
            assert r.status_code == 409
        finally:
            A.PIPELINE_LOCK.release()

    def test_delete_current_run_refused(self, client):
        rid = A.register_run("en")
        A.RUN_STATE["run_id"] = rid
        A.PIPELINE_LOCK.acquire()
        try:
            r = client.delete(f"/api/projects/{rid}")
            assert r.status_code == 409
        finally:
            A.RUN_STATE["run_id"] = None
            A.PIPELINE_LOCK.release()


# ------------------------------------------------------------- Fase 1 tests
def _write_graph(ws, payload):
    p = ws / "data" / "graph_out" / "graph_with_metrics.json"
    p.write_text(json.dumps(payload))
    return p


class TestGraphSanitization:
    def test_bidirectional_and_multi_relation_preserved(self, client, ws):
        _write_graph(ws, {
            "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "edges": [
                {"source": "a", "target": "b", "relation": "analyze"},
                {"source": "b", "target": "a", "relation": "influence"},  # direzione opposta
                {"source": "a", "target": "b", "relation": "co_occurs"},  # multi-relazione
                {"source": "a", "target": "a", "relation": "self"},       # self-loop: scartato
                {"source": "a", "target": "zzz", "relation": "dangling"}, # endpoint assente
            ],
        })
        r = client.get("/api/graph")
        assert r.status_code == 200
        edges = r.json()["edges"]
        pairs = [(e["source"], e["target"], e["relation"]) for e in edges]
        assert ("a", "b", "analyze") in pairs
        assert ("b", "a", "influence") in pairs, "direzione opposta persa"
        assert ("a", "b", "co_occurs") in pairs, "multi-relazione persa"
        assert all(s != t for s, t, _ in pairs)
        assert all(t in {"a", "b", "c"} and s in {"a", "b", "c"} for s, t, _ in pairs)


class TestRunRegistry:
    def _insert(self, status):
        conn = A.db()
        cur = conn.execute(
            "INSERT INTO runs (ts, lang, status, nodes, edges, graph_path) "
            "VALUES ('2026-07-28T00:00:00+00:00', 'en', ?, 0, 0, '')", (status,))
        conn.commit()
        rid = cur.lastrowid
        conn.close()
        return rid

    def test_error_runs_are_not_ghosted(self, client):
        rid_err = self._insert("error")
        rid_done = self._insert("done")  # done SENZA archivio = ghost da espungere
        r = client.get("/api/runs")
        ids = [run["id"] for run in r.json()["runs"]]
        assert rid_err in ids, "il run fallito deve restare visibile"
        assert rid_done not in ids, "il run done senza archivio e' un ghost"


# ------------------------------------------------- extract_svo (pure funcs)
import extract_svo as S


class TestAccentFolding:
    def test_normalize_folds_accents(self):
        assert S.normalize("società") == "societa"
        assert S.normalize("investigación") == "investigacion"
        assert S.normalize("perché") == "perche"
        # niente piu' split su accenti interni
        assert " " not in S.normalize("investigación")
        assert S.node_id_from(S.normalize("investigación")) == "investigacion"

    def test_reference_sets_folded(self):
        it_junk = S._fold_set(S.JUNK_ENTITIES_DICT["it"])
        assert "perche" in it_junk
        assert "cio" in it_junk
        es_heads = S._fold_set(S.GENERIC_HEADS_DICT["es"])
        assert "teoria" in es_heads
        assert "analisis" in es_heads

    def test_accented_junk_entity_rejected(self, monkeypatch):
        monkeypatch.setattr(S, "JUNK_ENTITIES",
                            S._fold_set(S.JUNK_ENTITIES_DICT["it"]))
        assert S.is_valid_entity("perché", "perche") is False
        assert S.is_valid_entity("critica sociale", "critica") is True

    def test_model_selection_env(self, monkeypatch):
        # auto -> lg ovunque; trf -> trf solo EN
        assert S.LANG_MODELS["en"]["auto"] == "en_core_web_lg"
        assert S.LANG_MODELS["en"]["trf"] == "en_core_web_trf"
        assert S.LANG_MODELS["it"]["trf"] == "it_core_news_lg"  # fallback documentato


class TestChunkText:
    def test_no_chunk_exceeds_max(self):
        text = "\n\n".join("parola " * 500 for _ in range(5))
        for chunk in S.chunk_text(text, max_chars=1800):
            assert len(chunk) <= 1800

    def test_hyphenated_linebreak_rejoined(self):
        out = S.chunk_text("soci-\nology rules", max_chars=5000)
        assert out == ["sociology rules"]


# -------------------------------------------------------- ingest_pdf naming
import ingest_pdf as I


class TestIngestNaming:
    def test_uppercase_pdf_produces_md(self, tmp_path):
        fitz = pytest.importorskip("fitz")
        pdf = tmp_path / "PAPER.PDF"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello deterministic world.")
        doc.save(str(pdf))
        doc.close()
        out_dir = tmp_path / "md"
        out_dir.mkdir()
        assert I.convert_pdf_classic(str(pdf), str(out_dir)) is True
        assert (out_dir / "PAPER.md").exists()
