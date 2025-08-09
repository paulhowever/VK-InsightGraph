import pandas as pd
import networkx as nx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_graph_from_messages(df, weight_col=None, export_metrics=False):
    """
    Build a directed graph from messages DataFrame and compute metrics if requested.
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        sender = row['sender']
        receiver = row['receiver']
        weight = row.get(weight_col, 1.0) if weight_col else 1.0
        if G.has_edge(sender, receiver):
            G[sender][receiver]['weight'] += weight
        else:
            G.add_edge(sender, receiver, weight=weight)
    
    if not export_metrics:
        return G
    
    # Compute metrics
    metrics = []
    nodes = list(G.nodes())
    if not nodes:
        return pd.DataFrame(columns=['user', 'degree', 'clustering', 'betweenness', 'eigenvector', 'is_isolated', 'is_bridge'])
    
    # Degree
    degrees = dict(G.degree(weight='weight'))
    
    # Clustering coefficient (using undirected graph for consistency)
    G_undirected = G.to_undirected()
    clustering = nx.clustering(G_undirected, weight='weight')
    
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    
    # Eigenvector centrality
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, weight='weight')
    except nx.NetworkXError:
        logger.warning("Eigenvector centrality failed to converge, setting to 0.0")
        eigenvector = {node: 0.0 for node in nodes}  # Fallback if convergence fails
    
    # Bridge detection (high betweenness)
    betweenness_threshold = sum(betweenness.values()) / len(betweenness) if betweenness else 0.0
    
    for node in nodes:
        metrics.append({
            'user': node,
            'degree': degrees.get(node, 0),
            'clustering': clustering.get(node, 0.0),
            'betweenness': betweenness.get(node, 0.0),
            'eigenvector': eigenvector.get(node, 0.0),
            'is_isolated': degrees.get(node, 0) <= 1,
            'is_bridge': betweenness.get(node, 0.0) > betweenness_threshold
        })
    
    metrics_df = pd.DataFrame(metrics)
    logger.info(f"Metrics computed: {metrics_df.columns.tolist()}")
    return metrics_df

if __name__ == "__main__":
    df = pd.read_csv("../mini_messages.csv", on_bad_lines='warn')
    G = build_graph_from_messages(df)
    metrics_df = build_graph_from_messages(df, export_metrics=True)
    print(metrics_df)
    metrics_df.to_csv("../data/user_metrics.csv", index=False)