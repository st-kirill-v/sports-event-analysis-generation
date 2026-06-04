from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go


def build_graph(edges: pd.DataFrame, team: str | None) -> nx.DiGraph:
    if team:
        edges = edges[edges["team_abbreviation"].str.upper() == team.upper()].copy()
    if edges.empty:
        raise ValueError("No edges found for the selected filters.")

    graph = nx.DiGraph()
    for row in edges.itertuples(index=False):
        passer = row.passer_name
        receiver = row.receiver_name
        weight = int(row.weight)
        graph.add_node(passer, player_id=int(row.passer_id), team=row.team_abbreviation)
        graph.add_node(receiver, player_id=int(row.receiver_id), team=row.team_abbreviation)
        if graph.has_edge(passer, receiver):
            graph[passer][receiver]["weight"] += weight
        else:
            graph.add_edge(passer, receiver, weight=weight)
    return graph


def centrality_table(graph: nx.DiGraph) -> pd.DataFrame:
    in_strength = dict(graph.in_degree(weight="weight"))
    out_strength = dict(graph.out_degree(weight="weight"))
    betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
    pagerank = nx.pagerank(graph, weight="weight")

    rows = []
    for node in graph.nodes:
        rows.append(
            {
                "player": node,
                "team": graph.nodes[node].get("team"),
                "in_strength": in_strength.get(node, 0),
                "out_strength": out_strength.get(node, 0),
                "betweenness": betweenness.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["pagerank", "out_strength"], ascending=[False, False])


def draw_graph(graph: nx.DiGraph, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=0.8)
    weights = [graph[u][v]["weight"] for u, v in graph.edges]
    max_weight = max(weights) if weights else 1
    node_sizes = [350 + graph.out_degree(node, weight="weight") * 55 for node in graph.nodes]
    edge_widths = [1 + 5 * weight / max_weight for weight in weights]

    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color="#4C78A8", alpha=0.9)
    nx.draw_networkx_edges(
        graph,
        pos,
        width=edge_widths,
        edge_color="#F58518",
        alpha=0.6,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.12",
    )
    nx.draw_networkx_labels(graph, pos, font_size=9, font_color="#111111")
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={(u, v): data["weight"] for u, v, data in graph.edges(data=True)},
        font_size=8,
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_interactive_graph(graph: nx.DiGraph, centrality: pd.DataFrame, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=0.8)
    centrality_by_player = centrality.set_index("player").to_dict(orient="index")
    weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    max_weight = max(weights) if weights else 1

    edge_traces = []
    edge_label_x = []
    edge_label_y = []
    edge_label_text = []
    edge_label_hover = []
    annotations = []

    for source, target, data in graph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        weight = int(data["weight"])
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line={"width": 1 + 5 * weight / max_weight, "color": "rgba(245,133,24,0.45)"},
                hoverinfo="text",
                text=f"{source} -> {target}<br>Assists: {weight}",
                showlegend=False,
            )
        )
        edge_label_x.append((x0 + x1) / 2)
        edge_label_y.append((y0 + y1) / 2)
        edge_label_text.append(str(weight))
        edge_label_hover.append(f"{source} -> {target}<br>Assists: {weight}")
        annotations.append(
            {
                "ax": x0,
                "ay": y0,
                "x": x1,
                "y": y1,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 3,
                "arrowsize": 1.2,
                "arrowwidth": 1,
                "arrowcolor": "rgba(245,133,24,0.35)",
                "opacity": 0.7,
            }
        )

    label_trace = go.Scatter(
        x=edge_label_x,
        y=edge_label_y,
        mode="text",
        text=edge_label_text,
        textfont={"size": 10, "color": "#5F370E"},
        hoverinfo="text",
        hovertext=edge_label_hover,
        showlegend=False,
    )

    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_size = []
    for node in graph.nodes:
        x, y = pos[node]
        stats = centrality_by_player.get(node, {})
        out_strength = int(stats.get("out_strength", 0))
        in_strength = int(stats.get("in_strength", 0))
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(18 + out_strength * 0.35)
        node_hover.append(
            f"{node}<br>"
            f"Team: {graph.nodes[node].get('team')}<br>"
            f"Assists given: {out_strength}<br>"
            f"Assists received: {in_strength}<br>"
            f"Betweenness: {float(stats.get('betweenness', 0.0)):.4f}<br>"
            f"PageRank: {float(stats.get('pagerank', 0.0)):.4f}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        hovertext=node_hover,
        marker={
            "size": node_size,
            "color": "#4C78A8",
            "line": {"width": 1.5, "color": "#1F2A35"},
            "opacity": 0.9,
        },
        showlegend=False,
    )

    fig = go.Figure(data=[*edge_traces, label_trace, node_trace])
    fig.update_layout(
        title=title,
        annotations=annotations,
        hovermode="closest",
        margin={"b": 20, "l": 20, "r": 20, "t": 60},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        plot_bgcolor="white",
        height=850,
    )
    fig.write_html(output_path, include_plotlyjs="cdn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a NetworkX assist graph from prepared edge data.")
    parser.add_argument("--edges", type=Path, default=Path("data/processed/assist_edges_weighted.csv"))
    parser.add_argument("--team", type=str, default="GSW", help="NBA team abbreviation, for example GSW or ATL.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/assist_graph"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(args.edges)
    graph = build_graph(edges, team=args.team)
    centrality = centrality_table(graph)

    suffix = args.team.upper() if args.team else "all_teams"
    centrality_path = args.output_dir / f"{suffix}_centrality.csv"
    graphml_path = args.output_dir / f"{suffix}_assist_graph.graphml"
    image_path = args.output_dir / f"{suffix}_assist_graph.png"
    html_path = args.output_dir / f"{suffix}_assist_graph.html"

    centrality.to_csv(centrality_path, index=False)
    nx.write_graphml(graph, graphml_path)
    draw_graph(graph, image_path, title=f"{suffix} assist network")
    write_interactive_graph(graph, centrality, html_path, title=f"{suffix} assist network")

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Centrality: {centrality_path}")
    print(f"GraphML: {graphml_path}")
    print(f"Image: {image_path}")
    print(f"Interactive HTML: {html_path}")
