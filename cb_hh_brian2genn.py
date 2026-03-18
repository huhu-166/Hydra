"""Brian2GeNN Morris-Lecar model for the current CB subnet export."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from brian2 import (
    NeuronGroup,
    SpikeMonitor,
    StateMonitor,
    Synapses,
    defaultclock,
    ms,
    mV,
    nA,
    nS,
    pF,
    prefs,
    run,
    second,
    set_device,
    siemens,
    start_scope,
    volt,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "hh_outputs"
NODES_PATH = Path(__file__).resolve().parent / "outputs" / "cb_nodes.csv"
EDGES_PATH = Path(__file__).resolve().parent / "outputs" / "cb_edges.csv"
SEED = 11
X_SPAN = 30.0
Y_SPAN = 60.0

C0 = 20.0 * pF
G_L0 = 2.0 * nS
G_NA0 = 4.4 * nS
G_K0 = 8.0 * nS
PHI0 = 0.04
E_L = -60.0 * mV
E_NA = 50.0 * mV
E_K = -84.0 * mV
V1 = -1.2 * mV
V2 = 18.0 * mV
V3 = 2.0 * mV
V4 = 30.0 * mV
TAU_W0 = 12.0 * ms

GROUP_SCALES = {
    "H": {"exc": 1.06, "rec": 0.2, "coupling": 0.76},
    "B1": {"exc": 1.14, "rec": 0.2, "coupling": 1.04},
    "B2": {"exc": 1.12, "rec": 0.2, "coupling": 1.04},
    "P": {"exc": 1.125, "rec": 0.17, "coupling": 1.04},
}

EDGE_TYPE_GAP = {
    "intra": 0.18 * nS,
    "feedforward": 0.9 * nS,
    "feedback": 0.14 * nS,
    "long_range": 0.05 * nS,
}

PAIR_SCALE = {
    ("H", "B1"): 3,
    ("H", "P"): 0.08,
    ("B1", "P"): 5.0,
    ("P", "P"): 3.6,
    ("P", "B2"): 5.0,
    ("B1", "B2"): 0.18,
    ("B2", "P"): 0.18,
    ("B2", "B2"): 3.6,
}


@dataclass
class SimulationSummary:
    backend: str
    n_neurons: int
    n_edges: int
    n_synapse_groups: int
    runtime_ms: float
    spike_count: int
    active_neurons: int
    group_total_counts: dict[str, int]
    group_active_counts: dict[str, int]
    mean_rate_hz: float
    h_first_spike_ms: float | None
    b1_first_spike_ms: float | None
    b2_first_spike_ms: float | None
    p_first_spike_ms: float | None
    detected_burst_count: int
    burst_centers_ms: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CB Morris-Lecar subnet in Brian2/Brian2GeNN.")
    parser.add_argument("--runtime-ms", type=float, default=4000.0)
    parser.add_argument("--dt-ms", type=float, default=0.02)
    parser.add_argument("--device", choices=["runtime", "genn"], default="genn")
    parser.add_argument("--genn-cpu", action="store_true")
    parser.add_argument("--trace-count-per-layer", type=int, default=3)
    parser.add_argument("--calcium-bin-ms", type=float, default=50.0)
    parser.add_argument("--calcium-tau-ms", type=float, default=300.0)
    parser.add_argument("--stim-amp-na", type=float, default=0.42)
    parser.add_argument("--stim-on-ms", type=float, default=0.0)
    parser.add_argument("--stim-off-ms", type=float, default=40.0)
    parser.add_argument("--stim-count", type=int, default=3)
    parser.add_argument("--genn-path", default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "latest"))
    return parser.parse_args()


def resolve_genn_path(cli_genn_path: str | None) -> Path:
    if cli_genn_path:
        candidates = [Path(cli_genn_path)]
    else:
        candidates = [Path("C:/Tools/genn-4.2.1"), Path("C:/Tools/genn-4.2.1/lib")]
        env_genn = os.environ.get("GENN_PATH")
        if env_genn:
            candidates.append(Path(env_genn))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "bin" / "genn-buildmodel.bat").exists():
            return resolved
        if (resolved / "lib" / "bin" / "genn-buildmodel.bat").exists():
            return resolved / "lib"
    raise FileNotFoundError("Could not find a compatible GeNN installation. Use --genn-path or set GENN_PATH to GeNN 4.2.1.")


def configure_device(device_name: str, output_dir: Path, genn_use_gpu: bool, cli_genn_path: str | None) -> str:
    if device_name == "runtime":
        prefs.codegen.target = "numpy"
        return "runtime"

    import brian2genn  # noqa: F401

    prefs.devices.genn.path = str(resolve_genn_path(cli_genn_path))
    if genn_use_gpu:
        prefs.devices.genn.cuda_backend.extra_compile_args_nvcc = ["-allow-unsupported-compiler"]
    set_device("genn", directory=str((output_dir / "genn_workspace").resolve()), use_GPU=genn_use_gpu)
    return "genn_gpu" if genn_use_gpu else "genn_cpu"


def load_subnet() -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(NODES_PATH).sort_values("node_id").reset_index(drop=True)
    edges = pd.read_csv(EDGES_PATH)
    return nodes, edges


def select_trace_ids(nodes: pd.DataFrame, per_group: int) -> list[int]:
    selected: list[int] = []
    for group in ("H", "B1", "B2", "P"):
        group_nodes = nodes[nodes["group"] == group].sort_values(["y", "x"]).head(per_group)
        selected.extend(group_nodes["node_id"].astype(int).tolist())
    return sorted(set(selected))


def select_stimulus_ids(nodes: pd.DataFrame, stim_count: int) -> np.ndarray:
    h_nodes = nodes[nodes["group"] == "H"].copy()
    target_x = X_SPAN / 2.0
    target_y = Y_SPAN - 1.0
    h_nodes["stim_distance"] = np.hypot(h_nodes["x"] - target_x, h_nodes["y"] - target_y)
    return h_nodes.sort_values("stim_distance").head(stim_count)["node_id"].to_numpy(dtype=int)


def build_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    dt_ms: float,
    trace_count_per_group: int,
    stim_amp_na: float,
    stim_on_ms: float,
    stim_off_ms: float,
    stim_count: int,
) -> tuple[NeuronGroup, list[Synapses], SpikeMonitor, StateMonitor, list[int], np.ndarray]:
    start_scope()
    rng = np.random.default_rng(SEED)
    defaultclock.dt = dt_ms * ms

    eqs = """
    dV/dt = (-gL*(V-EL) - gNa*minf*(V-ENa) - gK*W*(V-EK) + I_bias + I_ext + I_gap) / C : volt
    dW/dt = phi * (winf - W) / tauW : 1
    minf = 0.5 * (1.0 + tanh((V - v1_ml) / v2_ml)) : 1
    winf = 0.5 * (1.0 + tanh((V - v3_ml) / v4_ml)) : 1
    tauW = tau_w_base / cosh((V - v3_ml) / (2.0 * v4_ml)) : second
    stim_active = int((t >= stim_on) and (t < stim_off)) : 1
    I_ext = stim_amp * stim_active : amp
    I_gap : amp
    I_bias : amp
    stim_amp : amp
    gL : siemens
    gNa : siemens
    gK : siemens
    C : farad
    phi : 1
    tau_w_base : second
    EL : volt (constant)
    ENa : volt (constant)
    EK : volt (constant)
    v1_ml : volt (constant)
    v2_ml : volt (constant)
    v3_ml : volt (constant)
    v4_ml : volt (constant)
    """

    G = NeuronGroup(
        len(nodes),
        eqs,
        threshold="V > 0*mV",
        refractory=3.0 * ms,
        method="euler",
    )
    G.EL = E_L
    G.ENa = E_NA
    G.EK = E_K
    G.v1_ml = V1
    G.v2_ml = V2
    G.v3_ml = V3
    G.v4_ml = V4

    gL_vals = np.zeros(len(nodes))
    gNa_vals = np.zeros(len(nodes))
    gK_vals = np.zeros(len(nodes))
    c_vals = np.zeros(len(nodes))
    phi_vals = np.zeros(len(nodes))
    tau_w_vals = np.zeros(len(nodes))
    bias_vals = np.zeros(len(nodes))
    coupling_out = np.ones(len(nodes))
    coupling_in = np.ones(len(nodes))
    stim_amp_vals = np.zeros(len(nodes))

    for idx, row in nodes.iterrows():
        scales = GROUP_SCALES[row["group"]]
        exc_noise = rng.uniform(0.96, 1.04)
        rec_noise = rng.uniform(0.96, 1.04)
        gNa_vals[idx] = float((G_NA0 * scales["exc"] * exc_noise) / siemens)
        gL_vals[idx] = float((G_L0 * (2.0 - scales["exc"]) * rng.uniform(0.98, 1.04)) / siemens)
        gK_vals[idx] = float((G_K0 * rng.uniform(0.97, 1.03)) / siemens)
        c_vals[idx] = float(C0 / pF)
        phi_vals[idx] = PHI0 * scales["rec"] * rec_noise
        tau_w_vals[idx] = float(TAU_W0 / ms)
        bias_scale = scales["exc"]
        
        bias_vals[idx] = float((0.12 * nA * bias_scale * rng.uniform(0.94, 1.06)) / nA)
        coupling_out[idx] = scales["coupling"] * rng.uniform(0.94, 1.06)
        coupling_in[idx] = scales["coupling"] * rng.uniform(0.94, 1.06)

    stim_ids = select_stimulus_ids(nodes, stim_count=stim_count)
    if len(stim_ids):
        stim_amp_vals[stim_ids] = stim_amp_na
        bias_vals[stim_ids] = bias_vals[stim_ids] + 0.012 * stim_amp_na

    G.gL = gL_vals * siemens
    G.gNa = gNa_vals * siemens
    G.gK = gK_vals * siemens
    G.C = c_vals * pF
    G.phi = phi_vals
    G.tau_w_base = tau_w_vals * ms
    G.I_bias = bias_vals * nA
    G.stim_amp = stim_amp_vals * nA
    initial_v_mv = -52.0 + rng.normal(0.0, 3.0, len(nodes))
    initial_w = np.clip(rng.normal(0.18, 0.03, len(nodes)), 0.0, 1.0)
    b2_mask = nodes["group"].to_numpy() == "B2"
    initial_v_mv[b2_mask] = -55.3 + rng.normal(0.0, 1.2, int(b2_mask.sum()))
    initial_w[b2_mask] = np.clip(rng.normal(0.22, 0.02, int(b2_mask.sum())), 0.0, 1.0)
    G.V = initial_v_mv * mV
    G.W = initial_w
    G.I_gap = 0.0 * nA
    G.namespace["stim_on"] = stim_on_ms * ms
    G.namespace["stim_off"] = stim_off_ms * ms

    synapse_groups: list[Synapses] = []
    grouped_edges = edges.groupby(["source_group", "target_group", "delay_ms"], sort=True)
    for (source_group, target_group, delay_ms), group_df in grouped_edges:
        syn_name = f"S_{source_group}_{target_group}_d{int(delay_ms)}"
        S = Synapses(
            G,
            G,
            model="""
            w : siemens
            tau_lag : second (constant)
            dv_peer/dt = (V_pre - v_peer) / tau_lag : volt (clock-driven)
            I_gap_post = w * (v_peer - V_post) : amp (summed)
            """,
            method="euler",
            name=syn_name,
        )
        src = group_df["source"].to_numpy(dtype=int)
        dst = group_df["target"].to_numpy(dtype=int)
        S.connect(i=src, j=dst)
        S.v_peer = -52.0 * mV
        S.tau_lag = float(delay_ms) * ms

        base_weights = np.array([float(EDGE_TYPE_GAP[edge_type] / siemens) for edge_type in group_df["edge_type"]])
        src_scale = np.array([coupling_out[i] for i in src])
        dst_scale = np.array([coupling_in[j] for j in dst])
        topo_scale = group_df["weight"].to_numpy(dtype=float)
        pair_scale = PAIR_SCALE.get((source_group, target_group), 1.0)
        S.w = base_weights * src_scale * dst_scale * topo_scale * pair_scale * siemens
        synapse_groups.append(S)

    spike_monitor = SpikeMonitor(G)
    sample_ids = select_trace_ids(nodes, per_group=trace_count_per_group)
    state_monitor = StateMonitor(G, ["V", "W", "I_gap"], record=sample_ids)
    return G, synapse_groups, spike_monitor, state_monitor, sample_ids, stim_ids


def build_calcium_proxy(spike_times_ms: np.ndarray, runtime_ms: float, bin_ms: float, tau_ms: float, n_neurons: int) -> pd.DataFrame:
    bins = np.arange(0.0, runtime_ms + bin_ms, bin_ms)
    counts, edges = np.histogram(spike_times_ms, bins=bins)
    if len(counts) == 0:
        return pd.DataFrame({"time_ms": np.array([], dtype=float), "spike_rate_hz": np.array([], dtype=float), "calcium_proxy": np.array([], dtype=float)})
    times = edges[:-1] + 0.5 * bin_ms
    spike_rate_hz = counts * (1000.0 / bin_ms) / max(1, n_neurons)
    kernel_len = max(4, int(8 * tau_ms / bin_ms))
    kernel = np.exp(-(np.arange(kernel_len) * bin_ms) / tau_ms)
    kernel /= kernel.sum()
    calcium_proxy = np.convolve(counts.astype(float), kernel, mode="full")[: len(counts)]
    return pd.DataFrame({"time_ms": times, "spike_rate_hz": spike_rate_hz, "calcium_proxy": calcium_proxy})


def detect_bursts(calcium_df: pd.DataFrame) -> list[float]:
    if calcium_df.empty:
        return []
    signal = calcium_df["calcium_proxy"].to_numpy(dtype=float)
    if np.allclose(signal.max(), 0.0):
        return []
    threshold = signal.mean() + 1.2 * signal.std()
    burst_indices: list[int] = []
    refractory_bins = max(1, int(1200.0 / max(1.0, calcium_df["time_ms"].diff().median())))
    for idx in range(1, len(signal) - 1):
        if signal[idx] >= threshold and signal[idx] >= signal[idx - 1] and signal[idx] >= signal[idx + 1]:
            if burst_indices and idx - burst_indices[-1] < refractory_bins:
                if signal[idx] > signal[burst_indices[-1]]:
                    burst_indices[-1] = idx
            else:
                burst_indices.append(idx)
    return calcium_df.iloc[burst_indices]["time_ms"].round(2).tolist()


def first_spike_ms(spikes: pd.DataFrame, nodes: pd.DataFrame, group: str) -> float | None:
    ids = set(nodes[nodes["group"] == group]["node_id"].astype(int).tolist())
    subset = spikes[spikes["neuron_index"].isin(ids)]
    if subset.empty:
        return None
    return float(round(subset["spike_time_ms"].min(), 3))


def group_active_counts(spikes: pd.DataFrame, nodes: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    totals = {
        group: int(count)
        for group, count in nodes["group"].value_counts().sort_index().items()
    }
    if spikes.empty:
        active = {group: 0 for group in totals}
        return totals, active

    merged = spikes.merge(
        nodes[["node_id", "group"]],
        left_on="neuron_index",
        right_on="node_id",
        how="left",
    )
    active = {
        group: int(merged.loc[merged["group"] == group, "neuron_index"].nunique())
        for group in totals
    }
    return totals, active


def blend_with_white(color: str, blend: float) -> tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    mixed = (1.0 - blend) * base + blend * white
    return tuple(np.clip(mixed, 0.0, 1.0))


def save_outputs(
    output_dir: Path,
    summary: SimulationSummary,
    nodes: pd.DataFrame,
    sample_ids: list[int],
    stim_ids: np.ndarray,
    spike_monitor: SpikeMonitor,
    state_monitor: StateMonitor,
    calcium_bin_ms: float,
    calcium_tau_ms: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    spikes = pd.DataFrame({"neuron_index": np.asarray(spike_monitor.i[:], dtype=int), "spike_time_ms": np.asarray(spike_monitor.t[:] / ms, dtype=float)})
    spikes.to_csv(output_dir / "spikes.csv", index=False)

    calcium_df = build_calcium_proxy(
        spike_times_ms=spikes["spike_time_ms"].to_numpy(dtype=float) if not spikes.empty else np.array([], dtype=float),
        runtime_ms=summary.runtime_ms,
        bin_ms=calcium_bin_ms,
        tau_ms=calcium_tau_ms,
        n_neurons=summary.n_neurons,
    )
    calcium_df.to_csv(output_dir / "network_calcium_proxy.csv", index=False)

    sample_df = nodes.loc[nodes["node_id"].isin(sample_ids), ["node_id", "group", "region", "x", "y"]].copy()
    sample_df["stimulated"] = sample_df["node_id"].isin(stim_ids)
    sample_df.to_csv(output_dir / "sample_trace_neurons.csv", index=False)

    group_palette = {"H": "#006d77", "B1": "#4c956c", "B2": "#90be6d", "P": "#d62828"}

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 2]})
    if not calcium_df.empty:
        axes[0].plot(calcium_df["time_ms"] / 1000.0, calcium_df["calcium_proxy"], color="#d61f69", linewidth=2.0)
    axes[0].set_ylabel("Network Ca proxy")
    axes[0].set_title("CB network calcium proxy")

    if summary.spike_count:
        display_order = (
            nodes.sort_values(["y", "x"])
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "plot_index"})
            [["node_id", "group", "plot_index"]]
        )
        merged = spikes.merge(display_order, left_on="neuron_index", right_on="node_id", how="left")
        for group in ("H", "B1", "B2", "P"):
            subset = merged[merged["group"] == group]
            if subset.empty:
                continue
            axes[1].scatter(subset["spike_time_ms"] / 1000.0, subset["plot_index"], s=6, color=group_palette[group], alpha=0.78, label=group)
        axes[1].legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Neuron order (sorted by y)")
    axes[1].set_title("Long-term raster plot (peduncle at bottom)")
    fig.tight_layout()
    fig.savefig(output_dir / "raster.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    record_ids = list(state_monitor.record)
    node_lookup = nodes.set_index("node_id")
    group_counts_seen = {group: 0 for group in ("H", "B1", "B2", "P")}
    for trace_idx, neuron_id in enumerate(record_ids):
        group = node_lookup.loc[int(neuron_id), "group"]
        label = f"{group}:{int(neuron_id)}"
        shade_idx = group_counts_seen[group]
        group_counts_seen[group] += 1
        color = blend_with_white(group_palette[group], min(0.45, 0.14 * shade_idx))
        axes[0].plot(state_monitor.t / 1000.0 / ms, state_monitor.V[trace_idx] / mV, linewidth=1.3, color=color, label=label)
        axes[1].plot(state_monitor.t / 1000.0 / ms, state_monitor.W[trace_idx], linewidth=1.3, color=color, label=label)
        axes[2].plot(state_monitor.t / 1000.0 / ms, state_monitor.I_gap[trace_idx] / nA, linewidth=1.3, color=color, label=label)
    axes[0].set_ylabel("V (mV)")
    axes[1].set_ylabel("W")
    axes[2].set_ylabel("I_gap (nA)")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title("Sample Morris-Lecar traces")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "sample_traces.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    if not calcium_df.empty:
        ax.plot(calcium_df["time_ms"] / 1000.0, calcium_df["spike_rate_hz"], color="#2c3e50", linewidth=1.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean spike rate (Hz)")
    ax.set_title("Population firing rate")
    fig.tight_layout()
    fig.savefig(output_dir / "population_rate.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    backend = configure_device(args.device, output_dir, genn_use_gpu=not args.genn_cpu, cli_genn_path=args.genn_path)
    nodes, edges = load_subnet()
    _, synapse_groups, spike_monitor, state_monitor, sample_ids, stim_ids = build_network(
        nodes=nodes,
        edges=edges,
        dt_ms=args.dt_ms,
        trace_count_per_group=args.trace_count_per_layer,
        stim_amp_na=args.stim_amp_na,
        stim_on_ms=args.stim_on_ms,
        stim_off_ms=args.stim_off_ms,
        stim_count=args.stim_count,
    )
    run(args.runtime_ms * ms)

    spikes = pd.DataFrame({"neuron_index": np.asarray(spike_monitor.i[:], dtype=int), "spike_time_ms": np.asarray(spike_monitor.t[:] / ms, dtype=float)})
    calcium_df = build_calcium_proxy(
        spike_times_ms=spikes["spike_time_ms"].to_numpy(dtype=float) if not spikes.empty else np.array([], dtype=float),
        runtime_ms=args.runtime_ms,
        bin_ms=args.calcium_bin_ms,
        tau_ms=args.calcium_tau_ms,
        n_neurons=len(nodes),
    )
    burst_centers_ms = detect_bursts(calcium_df)
    spike_count = int(spike_monitor.num_spikes)
    active_neurons = int(len(np.unique(np.asarray(spike_monitor.i[:]))) if spike_count else 0)
    total_counts, active_counts = group_active_counts(spikes, nodes)
    summary = SimulationSummary(
        backend=backend,
        n_neurons=int(len(nodes)),
        n_edges=int(len(edges)),
        n_synapse_groups=len(synapse_groups),
        runtime_ms=args.runtime_ms,
        spike_count=spike_count,
        active_neurons=active_neurons,
        group_total_counts=total_counts,
        group_active_counts=active_counts,
        mean_rate_hz=(spike_count / len(nodes)) / (args.runtime_ms / 1000.0),
        h_first_spike_ms=first_spike_ms(spikes, nodes, "H"),
        b1_first_spike_ms=first_spike_ms(spikes, nodes, "B1"),
        b2_first_spike_ms=first_spike_ms(spikes, nodes, "B2"),
        p_first_spike_ms=first_spike_ms(spikes, nodes, "P"),
        detected_burst_count=len(burst_centers_ms),
        burst_centers_ms=burst_centers_ms,
    )
    save_outputs(
        output_dir=output_dir,
        summary=summary,
        nodes=nodes,
        sample_ids=sample_ids,
        stim_ids=stim_ids,
        spike_monitor=spike_monitor,
        state_monitor=state_monitor,
        calcium_bin_ms=args.calcium_bin_ms,
        calcium_tau_ms=args.calcium_tau_ms,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
