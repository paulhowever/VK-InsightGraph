import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_graph(G, metrics_df, users_to_plot, department_colors):
    if not G or not G.nodes():
        logger.warning("Graph is empty or None")
        return go.Figure()
    
    logger.info(f"Users to plot: {users_to_plot}")
    
    G = G.subgraph(users_to_plot).copy()
    if not G.nodes():
        logger.warning("No nodes to plot after subgraph filtering")
        return go.Figure()

    logger.info(f"Nodes in subgraph: {list(G.nodes())}")
    logger.info(f"Edges in subgraph: {list(G.edges(data=True))}")

    required_columns = ['user', 'department', 'degree', 'clustering', 'betweenness', 'group_id']
    if not all(col in metrics_df.columns for col in required_columns):
        logger.error(f"metrics_df missing required columns: {metrics_df.columns.tolist()}")
        return go.Figure()

    try:
        pos = nx.spring_layout(G, k=2.0 / np.sqrt(len(G.nodes())), iterations=100, seed=42)
        logger.info("Spring layout computed successfully")
    except Exception as e:
        logger.error(f"Error in spring_layout: {str(e)}")
        return go.Figure()

    node_x = []
    node_y = []
    node_colors = []
    node_outline_colors = []
    node_sizes = []
    node_text = []
    node_labels = []
    
    for node in G.nodes():
        try:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            dept = metrics_df[metrics_df['user'] == node]['department'].iloc[0] if node in metrics_df['user'].values else 'Unknown'
            node_colors.append(department_colors.get(dept, '#B0BEC5'))
            node_outline_colors.append(department_colors.get(dept, '#B0BEC5'))
            degree = metrics_df[metrics_df['user'] == node]['degree'].iloc[0] if node in metrics_df['user'].values else 0
            size = 20 + (degree * 5)
            node_sizes.append(min(40, size))
            metrics = metrics_df[metrics_df['user'] == node].iloc[0] if node in metrics_df['user'].values else {}
            text = (f"User: {node}<br>"
                    f"Department: {dept}<br>"
                    f"Degree: {metrics.get('degree', 0)}<br>"
                    f"Clustering: {metrics.get('clustering', 0):.2f}<br>"
                    f"Betweenness: {metrics.get('betweenness', 0):.2f}<br>"
                    f"Group: {metrics.get('group_id', 'N/A')}")
            node_text.append(text)
            node_labels.append(node if metrics.get('degree', 0) > 0 else '')
        except Exception as e:
            logger.error(f"Error processing node {node}: {str(e)}")
            continue

    if not node_x:
        logger.warning("No valid nodes to plot")
        return go.Figure()

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition='top center',
        textfont=dict(size=12, color='#000000', family='Arial'),
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            opacity=0.9,
            line=dict(width=3, color=node_outline_colors),
        ),
        hoverlabel=dict(bgcolor='#424242', font=dict(color='#FFFFFF', size=12))
    )

    annotations = []
    for edge in G.edges(data=True):
        try:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get('weight', 1.0)
            annotations.append(dict(
                ax=x0, ay=y0,
                x=x1, y=y1,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=max(1.0, min(2.0, weight * 0.5)),
                arrowwidth=max(3.0, min(6.0, weight * 1.0)),
                arrowcolor='#00BFFF',
                opacity=1.0
            ))
            logger.debug(f"Arrow added: {edge[0]} -> {edge[1]}, weight: {weight}, arrowwidth: {max(3.0, min(6.0, weight * 1.0))}")
        except Exception as e:
            logger.error(f"Error creating arrow for {edge}: {str(e)}")
            continue

    logger.info(f"Total arrows added: {len(annotations)}")
    if not annotations:
        logger.warning("No arrows were added, check subgraph or edge data")

    cluster_shapes = []
    group_ids = metrics_df[metrics_df['user'].isin(G.nodes())]['group_id'].unique()
    cluster_colors = ['rgba(100, 181, 246, 0.2)', 'rgba(255, 202, 40, 0.2)', 'rgba(76, 175, 80, 0.2)', 'rgba(239, 83, 80, 0.2)', 'rgba(171, 71, 188, 0.2)']
    for idx, group_id in enumerate(group_ids):
        group_nodes = metrics_df[metrics_df['group_id'] == group_id]['user'].tolist()
        group_nodes = [node for node in group_nodes if node in G.nodes()]
        if not group_nodes:
            continue
        x_coords = [pos[node][0] for node in group_nodes]
        y_coords = [pos[node][1] for node in group_nodes]
        if x_coords and y_coords:
            x_min, x_max = min(x_coords) - 0.1, max(x_coords) + 0.1
            y_min, y_max = min(y_coords) - 0.1, max(y_coords) + 0.1
            cluster_shapes.append(dict(
                type="rect",
                x0=x_min, y0=y_min, x1=x_max, y1=y_max,
                line=dict(color=cluster_colors[idx % len(cluster_colors)], width=2),
                fillcolor=cluster_colors[idx % len(cluster_colors)],
                opacity=0.2,
                layer='below'
            ))
            logger.debug(f"Cluster shape added for group {group_id}: x=({x_min}, {x_max}), y=({y_min}, {y_max})")

    logger.info(f"Total cluster shapes added: {len(cluster_shapes)}")

    legend_traces = []
    for dept, color in department_colors.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=color, line=dict(width=3, color=color)),
            name=dept
        ))
    for idx, group_id in enumerate(group_ids):
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=0, color=cluster_colors[idx % len(cluster_colors)], opacity=0.2),
            name=f'Группа {group_id}',
            showlegend=True
        ))

    fig = go.Figure(data=[node_trace] + legend_traces,
                    layout=go.Layout(
                        showlegend=True,
                        legend=dict(x=1.05, y=1, orientation='v', title='Отделы и группы', font=dict(color='#FFFFFF')),
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='#424242',
                        paper_bgcolor='#424242',
                        width=900,
                        height=700,
                        hoverdistance=20,
                        spikedistance=10,
                        annotations=annotations,
                        shapes=cluster_shapes
                    ))

    fig.update_traces(
        marker=dict(
            sizemode='diameter',
            sizeref=0.1,
            line=dict(width=3, color=node_outline_colors)
        ),
        selector=dict(mode='markers')
    )

    logger.info("Graph plotted successfully")
    return fig
