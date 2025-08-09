import pandas as pd
import networkx as nx
import community as community_louvain

from .graph_builder import build_graph_from_messages

def add_clusters_to_metrics(G, metrics_df):
    """
    Add Louvain community detection and 'is_overloaded' flag based on activity.
    """
    # Convert directed graph to undirected for Louvain
    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected, resolution=0.5, random_state=42)
    # Add +1 to group_id to start from 1
    group_df = pd.DataFrame([
        {'user': user, 'group_id': group_id + 1}
        for user, group_id in partition.items()
    ])
    
    # Debug: Log number of communities
    print(f"Найдено групп общения: {len(set(partition.values()))}")
    
    # Merge metrics and communities
    merged = metrics_df.merge(group_df, on='user')
    
    # Overloaded users — above 90th percentile of degree
    threshold = merged['degree'].quantile(0.9)
    merged['is_overloaded'] = merged['degree'] > threshold
    
    return merged

if __name__ == "__main__":
    df = pd.read_csv("../mini_messages.csv", on_bad_lines='warn')
    G = build_graph_from_messages(df)
    metrics_df = build_graph_from_messages(df, export_metrics=True)
    enriched_df = add_clusters_to_metrics(G, metrics_df)
    enriched_df.to_csv("../data/user_metrics_enriched.csv", index=False)
    print("✅ Группировка завершена. Файл сохранён: user_metrics_enriched.csv")