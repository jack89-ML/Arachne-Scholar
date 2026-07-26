from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Arachne Scholar Dashboard")

# Monta la cartella static per HTML/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/graph")
def get_graph():
    # Cerca prima il grafo con le metriche SNA, altrimenti quello base
    metrics_path = "../graph_out/graph_with_metrics.json"
    base_path = "../graph_out/graph.json"
    
    if os.path.exists(metrics_path):
        return FileResponse(metrics_path)
    elif os.path.exists(base_path):
        return FileResponse(base_path)
    return {"error": "Grafo non trovato. Esegui prima la pipeline."}
