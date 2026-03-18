"""Build and visualize a biologically-guided Hydra CB subnet.

The 2D plane uses x as circumference (0..30) and y as body-axis height (0..60),
with peduncle at y=0 and hypostome toward y=60.
The 3D view rolls the x direction into a cylinder and keeps y as cylinder height.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 7

X_SPAN = 30.0
Y_SPAN = 60.0
OPEN_CYLINDER_ANGLE = 2.0 * math.pi
CYLINDER_RADIUS = X_SPAN / OPEN_CYLINDER_ANGLE
X_SAMPLE_MIN = 3.0
X_SAMPLE_MAX = 27.0

GROUP_SPECS = {
    "H": {
        "count": 22,
        "label": "hypostome integration",
        "region": "hypostome",
        "color": "#006d77",
        "y_range": (50.0, 58.0),
    },
    "B1": {
        "count": 40,
        "label": "body column relay",
        "region": "column",
        "color": "#4c956c",
        "y_range": (8.0, 52.0),
    },
    "B2": {
        "count": 60,
        "label": "body CB distributed",
        "region": "column",
        "color": "#90be6d",
        "y_range": (8.0, 44.0),
    },
    "P": {
        "count": 52,
        "label": "peduncle motor core",
        "region": "peduncle",
        "color": "#d62828",
        "y_range": (0.0, 12.0),
    },
}

CONNECTION_RULES = {
    ("H", "H"): {"prob": 0.2, "weight": 0.92, "edge_type": "intra"},
    ("B1", "B1"): {"prob": 0.15, "weight": 0.94, "edge_type": "intra"},
    ("B2", "B2"): {"prob": 0.26, "weight": 1.02, "edge_type": "intra"},
    ("P", "P"): {"prob": 0.260, "weight": 1.28, "edge_type": "intra"},
    ("H", "B1"): {"prob": 0.28, "weight": 1.10, "edge_type": "feedforward"},
    ("B1", "B2"): {"prob": 0.007, "weight": 0.88, "edge_type": "feedback"},
    ("B2", "P"): {"prob": 0.03, "weight": 0.96, "edge_type": "feedback"},
    ("H", "P"): {"prob": 0.001, "weight": 0.90, "edge_type": "long_range"},
    ("B1", "P"): {"prob": 0.28, "weight": 1.05, "edge_type": "feedforward"},
    ("P", "B2"): {"prob": 0.30, "weight": 0.90, "edge_type": "feedforward"},
    ("B2", "B1"): {"prob": 0.028, "weight": 0.55, "edge_type": "feedback"},
    ("P", "H"): {"prob": 0.010, "weight": 0.50, "edge_type": "feedback"},
    ("B1", "H"): {"prob": 0.008, "weight": 0.55, "edge_type": "feedback"},
}

LOCAL_LENGTH_SCALE = {
    "H": 6.5,
    "B1": 8.0,
    "B2": 8.0,
    "P": 6.0,
}


@dataclass
class Node:
    node_id: int
    group: str
    region: str
    x: float
    y: float
    theta: float
    x3d: float
    y3d: float
    z3d: float


def planar_distance(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def delay_from_distance(distance_value: float) -> int:
    if distance_value < 4.0:
        return 1
    if distance_value < 10.0:
        return 2
    if distance_value < 20.0:
        return 3
    return 4


def sample_positions(rng: np.random.Generator) -> list[Node]:
    nodes: list[Node] = []
    node_id = 0

    def stratified_y(count: int, y_low: float, y_high: float, jitter: float) -> np.ndarray:
        base = np.linspace(y_low + 0.35, y_high - 0.35, count)
        noise = rng.uniform(-jitter, jitter, count)
        return np.clip(base + noise, y_low + 0.2, y_high - 0.2)

    for group_name in ("H", "B1", "B2", "P"):
        spec = GROUP_SPECS[group_name]
        y_low, y_high = spec["y_range"]
        if group_name in {"H", "B1", "B2"}:
            if group_name == "H":
                jitter = 0.35
            elif group_name == "B1":
                jitter = 0.42
            else:
                jitter = 0.55
            y_values = stratified_y(spec["count"], y_low, y_high, jitter=jitter)
        else:
            y_values = rng.uniform(y_low + 0.2, y_high - 0.2, spec["count"])
        x_values = rng.uniform(X_SAMPLE_MIN, X_SAMPLE_MAX, spec["count"])
        for x_value, y_value in zip(x_values, y_values):
            x = float(x_value)
            y = float(y_value)
            theta = (x / X_SPAN) * OPEN_CYLINDER_ANGLE - OPEN_CYLINDER_ANGLE / 2.0
            x3d = CYLINDER_RADIUS * math.cos(theta)
            y3d = CYLINDER_RADIUS * math.sin(theta)
            z3d = y
            nodes.append(
                Node(
                    node_id=node_id,
                    group=group_name,
                    region=spec["region"],
                    x=x,
                    y=y,
                    theta=theta,
                    x3d=x3d,
                    y3d=y3d,
                    z3d=z3d,
                )
            )
            node_id += 1
    return nodes


def add_edge(edges: list[dict], source: Node, target: Node, weight: float, edge_type: str) -> None:
    distance_value = planar_distance(source, target)
    edges.append(
        {
            "source": source.node_id,
            "target": target.node_id,
            "source_group": source.group,
            "target_group": target.group,
            "source_region": source.region,
            "target_region": target.region,
            "distance": distance_value,
            "delay_ms": delay_from_distance(distance_value),
            "weight": weight,
            "edge_type": edge_type,
        }
    )


def build_edges(nodes: list[Node], rng: np.random.Generator) -> list[dict]:
    group_to_nodes: dict[str, list[Node]] = {}
    for node in nodes:
        group_to_nodes.setdefault(node.group, []).append(node)

    edges: list[dict] = []
    seen: set[tuple[int, int]] = set()

    def distance_factor(source: Node, target: Node) -> float:
        if source.group == target.group:
            lam = LOCAL_LENGTH_SCALE[source.group]
            return math.exp(-planar_distance(source, target) / lam)
        if {source.group, target.group} <= {"B1", "B2"}:
            return math.exp(-planar_distance(source, target) / 9.5)
        if {source.group, target.group} <= {"B2", "P"}:
            if source.group == "P" and target.group == "B2":
                return math.exp(-planar_distance(source, target) / 4.8)
            return math.exp(-planar_distance(source, target) / 7.2)
        if source.group == "B1" and target.group == "P":
            return math.exp(-planar_distance(source, target) / 5.2)
        if {source.group, target.group} <= {"H", "B1"}:
            return math.exp(-planar_distance(source, target) / 8.8)
        return math.exp(-planar_distance(source, target) / 6.5)

    def maybe_connect(source: Node, target: Node, rule: dict[str, float]) -> None:
        if source.node_id == target.node_id:
            return
        key = (source.node_id, target.node_id)
        if key in seen:
            return
        if source.group == "B1" and target.group == "P":
            if planar_distance(source, target) > 16.0:
                return
            if abs(source.y - target.y) > 15.0:
                return
        if source.group == "P" and target.group == "B2":
            if planar_distance(source, target) > 12.0:
                return
            if abs(source.y - target.y) > 10.0:
                return
        probability = rule["prob"] * distance_factor(source, target)
        probability = min(1.0, probability)
        if rng.random() < probability:
            add_edge(edges, source, target, rule["weight"], rule["edge_type"])
            seen.add(key)

    for (source_group, target_group), rule in CONNECTION_RULES.items():
        for source in group_to_nodes[source_group]:
            for target in group_to_nodes[target_group]:
                maybe_connect(source, target, rule)

    return edges


def edge_color() -> dict[str, str]:
    return {
        "intra": "#ff9f1c",
        "feedforward": "#3a86ff",
        "feedback": "#d90429",
        "long_range": "#8338ec",
    }


def draw_2d(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 12.5))
    region_fill = [
        (0.0, 15.0, "#fde2e4", "peduncle"),
        (15.0, 52.0, "#edf6f9", "body column"),
        (52.0, 60.0, "#e0fbfc", "hypostome"),
    ]
    for y0, y1, color, _ in region_fill:
        ax.axhspan(y0, y1, color=color, alpha=0.9, zorder=0)

    palette = edge_color()
    for edge in edges_df.itertuples(index=False):
        src = nodes_df.iloc[edge.source]
        dst = nodes_df.iloc[edge.target]
        ax.plot([src.x, dst.x], [src.y, dst.y], color=palette[edge.edge_type], alpha=0.16, linewidth=1.2, zorder=1)

    for group_name in ("H", "B1", "B2", "P"):
        spec = GROUP_SPECS[group_name]
        group_df = nodes_df[nodes_df["group"] == group_name]
        ax.scatter(group_df["x"], group_df["y"], s=28, color=spec["color"], label=f"{group_name} {spec['label']}", zorder=2)

    ax.set_xlim(0.0, X_SPAN)
    ax.set_ylim(0.0, Y_SPAN)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.linspace(0.0, X_SPAN, 7))
    ax.set_yticks(np.linspace(0.0, Y_SPAN, 7))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("x (circumference / transverse axis)")
    ax.set_ylabel("y (body axis / cylinder height)")
    ax.set_title("Biology-guided Hydra CB subnet", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=8, frameon=True)
    fig.subplots_adjust(top=0.83, left=0.07, right=0.98, bottom=0.10)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def draw_3d(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(11.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    palette = edge_color()

    def roll_point(x_value: float, y_value: float) -> tuple[float, float, float]:
        theta = (x_value / X_SPAN) * OPEN_CYLINDER_ANGLE - OPEN_CYLINDER_ANGLE / 2.0
        return (
            CYLINDER_RADIUS * math.cos(theta),
            CYLINDER_RADIUS * math.sin(theta),
            y_value,
        )

    for edge in edges_df.itertuples(index=False):
        src = nodes_df.iloc[edge.source]
        dst = nodes_df.iloc[edge.target]
        t_values = np.linspace(0.0, 1.0, 20)
        x_values = (1.0 - t_values) * src.x + t_values * dst.x
        y_values = (1.0 - t_values) * src.y + t_values * dst.y
        rolled = np.array([roll_point(float(xv), float(yv)) for xv, yv in zip(x_values, y_values)])
        ax.plot(rolled[:, 0], rolled[:, 1], rolled[:, 2], color=palette[edge.edge_type], alpha=0.16, linewidth=1.2)

    for group_name in ("H", "B1", "B2", "P"):
        spec = GROUP_SPECS[group_name]
        group_df = nodes_df[nodes_df["group"] == group_name]
        ax.scatter(group_df["x3d"], group_df["y3d"], group_df["z3d"], s=18, color=spec["color"], label=group_name)

    ax.set_box_aspect((2 * CYLINDER_RADIUS, 2 * CYLINDER_RADIUS, Y_SPAN))
    ax.set_xlim(-CYLINDER_RADIUS, CYLINDER_RADIUS)
    ax.set_ylim(-CYLINDER_RADIUS, CYLINDER_RADIUS)
    ax.set_zlim(0.0, Y_SPAN)
    ax.set_xticks([-CYLINDER_RADIUS, 0.0, CYLINDER_RADIUS])
    ax.set_yticks([-CYLINDER_RADIUS, 0.0, CYLINDER_RADIUS])
    ax.set_zticks(np.linspace(0.0, Y_SPAN, 7))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel("cylinder x")
    ax.set_ylabel("cylinder y")
    ax.set_zlabel("")
    ax.set_title("CB subnet rolled into a cylinder", pad=16)
    ax.text2D(0.02, 0.52, "height (y)", transform=ax.transAxes, rotation=90, va="center", ha="left", fontsize=10)
    ax.view_init(elev=18, azim=-58)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=8, frameon=True)
    fig.subplots_adjust(top=0.84, left=0.02, right=0.98, bottom=0.04)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, output_path: Path) -> None:
    summary = {
        "seed": SEED,
        "n_neurons": int(len(nodes_df)),
        "n_edges": int(len(edges_df)),
        "group_counts": nodes_df["group"].value_counts().sort_index().to_dict(),
        "region_counts": nodes_df["region"].value_counts().sort_index().to_dict(),
        "edge_type_counts": edges_df["edge_type"].value_counts().sort_index().to_dict(),
        "pair_counts": {
            f"{source}->{target}": int(count)
            for (source, target), count in edges_df.groupby(["source_group", "target_group"]).size().to_dict().items()
        },
        "delay_counts_ms": edges_df["delay_ms"].value_counts().sort_index().to_dict(),
        "mean_out_degree": float(edges_df.groupby("source").size().mean()),
        "mean_in_degree": float(edges_df.groupby("target").size().mean()),
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    nodes = sample_positions(rng)
    nodes_df = pd.DataFrame(asdict(node) for node in nodes).sort_values("node_id").reset_index(drop=True)
    edges_df = pd.DataFrame(build_edges(nodes, rng))

    nodes_df.to_csv(OUTPUT_DIR / "cb_nodes.csv", index=False)
    edges_df.to_csv(OUTPUT_DIR / "cb_edges.csv", index=False)
    write_summary(nodes_df, edges_df, OUTPUT_DIR / "cb_summary.json")
    draw_2d(nodes_df, edges_df, OUTPUT_DIR / "cb_network_2d.png")
    draw_3d(nodes_df, edges_df, OUTPUT_DIR / "cb_network_3d.png")


if __name__ == "__main__":
    main()
