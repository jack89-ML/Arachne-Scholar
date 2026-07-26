import json, os, sys
import networkx as nx
import networkx.algorithms.community as nx_comm

def calculate_metrics(graph_path, out_path, sample_k=500, top_n=200):
    print(f"Caricamento grafo da: {graph_path}")
    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node in data["nodes"]:
        G.add_node(node["id"], type=node.get("type", "concept"), label=node.get("label", node["id"]))
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""), weight=edge.get("weight", 1))

    G_undirected = G.to_undirected()
    print(f"Grafo in memoria: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi.")

    # Degree centrality su tutto il grafo (costo lineare)
    degrees = dict(G.degree())

    # --- v1.4: BURT CONSTRAINT AMPUTATO -------------------------------------
    # nx.constraint e' O(somma deg^2): con 84k+ archi e hub da 2000+ gradi
    # cicla per decine di minuti in Python puro. Rimosso per decisione di
    # progetto: il campo resta nello schema JSON (default 1.0) per non
    # rompere il contratto col frontend.
    top_nodes = sorted(degrees, key=lambda n: degrees[n], reverse=True)[:top_n]
    top_set = set(top_nodes)
    print(f"Top {len(top_nodes)} hub per degree selezionati (su {G.number_of_nodes()} nodi).")

    # Betweenness ESATTA sul sottografo indotto dai top hub (filtro diretto).
    G_sub = G_undirected.subgraph(top_nodes).copy()
    print(f"Calcolo Betweenness Centrality (sottografo hub: {G_sub.number_of_nodes()} nodi, "
          f"{G_sub.number_of_edges()} archi)...")
    betweenness = nx.betweenness_centrality(G_sub, weight=None)

    # Community Detection sul grafo completo (Louvain, lineare-ish).
    print("Calcolo Community Detection (Louvain) sul grafo completo...")
    communities = nx_comm.louvain_communities(G_undirected, seed=42)
    comm_map = {node: i for i, comm in enumerate(communities) for node in comm}

    for node in data["nodes"]:
        nid = node["id"]
        b_val = round(betweenness.get(nid, 0), 4) if nid in top_set else 0.0
        node["metrics"] = {
            "degree": degrees.get(nid, 0),
            "betweenness": b_val,
            "constraint": 1.0,  # Burt amputato: default neutro, schema invariato
            "community": comm_map.get(nid, 0)
        }

    data["meta"] = {**data.get("meta", {}),
                    "sna_mode": f"top{top_n}-hub-noburt",
                    "sna_note": ("Burt constraint amputato (incompatibile con la densita' del grafo): "
                                 "betweenness esatta su sottografo indotto top-N hub; "
                                 "constraint=1.0 default per tutti i nodi; degree e Louvain full graph.")}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Metriche SNA salvate in {out_path}")

if __name__ == "__main__":
    calculate_metrics(
        sys.argv[1] if len(sys.argv) > 1 else "../graph_out/graph.json",
        sys.argv[2] if len(sys.argv) > 2 else "../graph_out/graph_with_metrics.json",
        int(sys.argv[3]) if len(sys.argv) > 3 else 500,
        int(sys.argv[4]) if len(sys.argv) > 4 else 200,
    )
