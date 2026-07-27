#!/usr/bin/env python3
"""Arachne Scholar -- RESET NUCLEARE.

Svuota completamente la tabella `runs` di SQLite e pialla tutto il contenuto
di data/graph_out/runs/, oltre ai file live (graph.json,
graph_with_metrics.json, active_run.txt).

Idempotente: eseguirlo N volte produce sempre lo stesso stato immacolato.
Nessun vecchio progetto sopravvive.

Uso:
    python3 reset_db.py
"""
import os
import shutil
import sqlite3

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(DATA_DIR, "graph_out")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
DB_PATH = os.path.join(DATA_DIR, "arachne.db")

LIVE_FILES = ("graph.json", "graph_with_metrics.json", "active_run.txt")


def main():
    # 1) DB: svuota la tabella runs e riallinea la sequence AUTOINCREMENT
    n_db = 0
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        try:
            n_db = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            conn.execute("DELETE FROM runs")
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name='runs'")
            except sqlite3.OperationalError:
                pass  # tabella sequence assente: non critico
            conn.commit()
        finally:
            conn.close()

    # 2) Pialla il contenuto di runs/ (mantiene la cartella)
    n_files = 0
    if os.path.isdir(RUNS_DIR):
        for name in os.listdir(RUNS_DIR):
            p = os.path.join(RUNS_DIR, name)
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    os.remove(p)
                except OSError:
                    pass
            n_files += 1
    else:
        os.makedirs(RUNS_DIR, exist_ok=True)

    # 3) Rimuovi i file live del grafo attivo
    n_live = 0
    for fname in LIVE_FILES:
        p = os.path.join(OUT_DIR, fname)
        if os.path.exists(p):
            try:
                os.remove(p)
                n_live += 1
            except OSError:
                pass

    print(f"[reset_db] OK -- record 'runs' eliminati: {n_db}; "
          f"artefatti runs/ eliminati: {n_files}; file live rimossi: {n_live}")
    print("[reset_db] DB immacolato: nessun progetto sopravvissuto.")


if __name__ == "__main__":
    main()
