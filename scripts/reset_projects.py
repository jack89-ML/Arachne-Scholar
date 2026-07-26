#!/usr/bin/env python3
"""Arachne-Scholar — Reset totale dei progetti (migrazione una tantum).

Svuota in modo definitivo:
  1. Tabella `runs` del database SQLite (con reset della sequenza ID).
  2. Archivi grafo in data/graph_out/runs/.
  3. File live (graph.json, graph_with_metrics.json) e marker active_run.txt.

Uso:
    python3 scripts/reset_projects.py --yes

Richiede --yes come conferma esplicita. Solo libreria standard.
"""
import os
import shutil
import sqlite3
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(DATA_DIR, "graph_out")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
DB_PATH = os.path.join(DATA_DIR, "arachne.db")


def main():
    if "--yes" not in sys.argv:
        print("RESET TOTALE PROGETTI — operazione distruttiva e irreversibile.")
        print("Rilancia con:  python3 scripts/reset_projects.py --yes")
        sys.exit(1)

    print(f"[reset] workspace: {BASE_DIR}")

    # 1. Database SQLite ----------------------------------------------------
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        try:
            before = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            conn.execute("DELETE FROM runs")
            try:
                conn.execute("DELETE FROM sqlite_sequence WHERE name='runs'")
            except sqlite3.OperationalError:
                pass  # tabella sqlite_sequence assente: nessun autoincremento da resettare
            conn.commit()
            after = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            print(f"[reset] DB: {before} record eliminati da 'runs' "
                  f"(rimasti: {after}), sequenza ID azzerata")
        finally:
            conn.close()
    else:
        print("[reset] DB: arachne.db assente, nulla da pulire")

    # 2. Archivi grafo -------------------------------------------------------
    removed = 0
    if os.path.isdir(RUNS_DIR):
        for name in os.listdir(RUNS_DIR):
            p = os.path.join(RUNS_DIR, name)
            try:
                if os.path.isdir(p) and not os.path.islink(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)
                removed += 1
            except OSError as exc:
                print(f"[reset] ATTENZIONE: impossibile rimuovere {p}: {exc}")
    print(f"[reset] FS: {removed} archivi eliminati da runs/")

    # 3. File live + marker attivo -------------------------------------------
    for fname in ("graph.json", "graph_with_metrics.json", "active_run.txt"):
        p = os.path.join(OUT_DIR, fname)
        if os.path.exists(p):
            os.remove(p)
            print(f"[reset] FS: rimosso file live {fname}")

    print("[reset] COMPLETATO — dashboard riportata allo stato 'vuoto pulito'")


if __name__ == "__main__":
    main()
