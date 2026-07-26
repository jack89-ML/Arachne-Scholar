import json
import os
import sys
import networkx as nx

def calculate_metrics(graph_path, out_path):
    print(f"Caricamento grafo da: {graph_path}")
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Costruiamo il grafo diretto con NetworkX
    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"], type=node.get("type", "concept"), label=node.get("label", node["id"]))
    
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""), weight=edge.get("weight", 1))

    print(f"Grafo in memoria: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi.")

    # 1. Betweenness Centrality (Concetti Ponte)
    print("Calcolo Betweenness Centrality...")
    betweenness = nx.betweenness_centrality(G, weight=None)
    
    # 2. Structural Holes (Burt's Constraint)
    # Più il constraint è BASSO, più il nodo copre un buco strutturale.
    print("Calcolo Buchi Strutturali (Constraint)...")
    # Il constraint lavora meglio su reti non dirette per la topologia di base
    G_undirected = G.to_undirected()
    constraint = nx.constraint(G_undirected)

    # Arricchiamo i nodi originali con le metriche SNA
    for node in data["nodes"]:
        nid = node["id"]
        node["metrics"] = {
            "degree": G.degree(nid),
            "betweenness": round(betweenness.get(nid, 0), 4),
            "constraint": round(constraint.get(nid, 0), 4) if nid in constraint and not str(constraint.get(nid, 0)) == 'nan' else 1.0
        }

    # Salviamo il JSON aggiornato
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Metriche SNA calcolate e salvate in {out_path}")

if __name__ == "__main__":
    in_file = sys.argv[1] if len(sys.argv) > 1 else "../graph_out/graph.json"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "../graph_out/graph_with_metrics.json"
    if os.path.exists(in_file):
        calculate_metrics(in_file, out_file)
    else:
        print(f"Errore: file {in_file} non trovato.")
