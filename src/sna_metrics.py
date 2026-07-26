import json, os, sys
import networkx as nx
import networkx.algorithms.community as nx_comm

def calculate_metrics(graph_path, out_path, sample_k=500):
    print(f"Caricamento grafo da: {graph_path}")
    with open(graph_path, "r", encoding="utf-8") as f: data = json.load(f)
    
    G = nx.DiGraph()
    for node in data["nodes"]: G.add_node(node["id"], type=node.get("type", "concept"), label=node.get("label", node["id"]))
    for edge in data["edges"]: G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""), weight=edge.get("weight", 1))

    G_undirected = G.to_undirected()
    
    print(f"Calcolo Betweenness Centrality (k={min(sample_k, G.number_of_nodes())})...")
    betweenness = nx.betweenness_centrality(G, k=min(sample_k, G.number_of_nodes()), weight=None)
    
    print("Calcolo Buchi Strutturali (Constraint)...")
    constraint = nx.constraint(G_undirected)

    print("Calcolo Community Detection (Louvain)...")
    communities = nx_comm.louvain_communities(G_undirected, seed=42)
    comm_map = {node: i for i, comm in enumerate(communities) for node in comm}

    for node in data["nodes"]:
        nid = node["id"]
        node["metrics"] = {
            "degree": G.degree(nid),
            "betweenness": round(betweenness.get(nid, 0), 4),
            "constraint": round(constraint.get(nid, 0), 4) if nid in constraint and not str(constraint.get(nid, 0)) == 'nan' else 1.0,
            "community": comm_map.get(nid, 0)
        }

    with open(out_path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Metriche SNA salvate in {out_path}")

if __name__ == "__main__":
    calculate_metrics(
        sys.argv[1] if len(sys.argv) > 1 else "../graph_out/graph.json",
        sys.argv[2] if len(sys.argv) > 2 else "../graph_out/graph_with_metrics.json",
        int(sys.argv[3]) if len(sys.argv) > 3 else 500
    )
