import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\ASUS\Documents\Hydra\Hydra_download_videos\mechanosensory\fig4"
OUT_DIR = r"C:\Users\ASUS\Documents\Hydra\Hydra_download_videos\mechanosensory"
HYDRA_IDS = ["Hydra1", "Hydra2", "Hydra3"]

# Fixed oral neurons: rows 1,2,3 (1-based) in NeuronROI_RawFluorAllFrames excluding peduncle row
ORAL_ROWS_1BASED = [1, 2, 3]

FPS = 16.67
TOTAL_MIN = 30
SPONT_MIN = 20
STIM_MIN = 10
STIM_ON_SEC = 1
STIM_OFF_SEC = 30

APPLY_SMOOTHING = True
SG_ORDER = 3
SG_FRAME = 11


def compute_dff_cummin(raw: np.ndarray) -> np.ndarray:
    f0 = np.minimum.accumulate(raw)
    f0 = np.maximum(f0, np.finfo(np.float64).eps)
    return (raw - f0) / f0


def maybe_smooth(y: np.ndarray) -> np.ndarray:
    if not APPLY_SMOOTHING:
        return y
    n = len(y)
    frame = SG_FRAME
    if frame <= SG_ORDER:
        frame = SG_ORDER + 2
    if frame % 2 == 0:
        frame += 1
    if frame > n:
        frame = n if n % 2 == 1 else n - 1
    if n >= SG_ORDER + 2 and frame > SG_ORDER:
        return savgol_filter(y, frame, SG_ORDER, mode="interp")
    return y


def build_protocol(n_frames: int, fps: float):
    """Protocol: 20 min off, then 10 min repeated 1s on / 30s off."""
    stim = np.zeros(n_frames, dtype=np.int8)
    start_stim = int(round(SPONT_MIN * 60 * fps))
    end_stim = min(n_frames, int(round((SPONT_MIN + STIM_MIN) * 60 * fps)))
    cycle = STIM_ON_SEC + STIM_OFF_SEC

    t_sec = np.arange(n_frames) / fps
    onset_frames = []
    onset = start_stim
    while onset < end_stim:
        on_end = min(end_stim, onset + int(round(STIM_ON_SEC * fps)))
        stim[onset:on_end] = 1
        onset_frames.append(onset)
        onset += int(round(cycle * fps))

    return stim, np.array(onset_frames, dtype=np.int64), t_sec


os.makedirs(OUT_DIR, exist_ok=True)

all_rows = []
meta_rows = []

oral_idx0 = [i - 1 for i in ORAL_ROWS_1BASED]

for hid in HYDRA_IDS:
    mat_path = os.path.join(BASE_DIR, hid, "IndividualNeuronTraces.mat")
    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    raw_all = np.asarray(d["NeuronROI_RawFluorAllFrames"], dtype=np.float64)
    # first N-1 rows are neurons, last row is peduncle ROI mean fluorescence
    neuron_raw = raw_all[:-1, :]
    ped_raw = raw_all[-1, :]

    n_neurons, n_frames = neuron_raw.shape
    if max(oral_idx0) >= n_neurons:
        raise ValueError(f"{hid}: oral row index out of range. neurons={n_neurons}")

    # dF/F
    n1 = maybe_smooth(compute_dff_cummin(neuron_raw[oral_idx0[0], :]))
    n2 = maybe_smooth(compute_dff_cummin(neuron_raw[oral_idx0[1], :]))
    n3 = maybe_smooth(compute_dff_cummin(neuron_raw[oral_idx0[2], :]))
    oral_mean = (n1 + n2 + n3) / 3.0
    ped = maybe_smooth(compute_dff_cummin(ped_raw))

    # limit to first 30 minutes if recording longer
    n_use = min(n_frames, int(round(TOTAL_MIN * 60 * FPS)))
    frame = np.arange(1, n_use + 1, dtype=np.int32)
    time_sec = (frame - 1) / FPS
    time_min = time_sec / 60.0

    n1, n2, n3 = n1[:n_use], n2[:n_use], n3[:n_use]
    oral_mean = oral_mean[:n_use]
    ped = ped[:n_use]

    stim_state, onset_frames, _ = build_protocol(n_use, FPS)

    df = pd.DataFrame({
        "HydraID": hid,
        "Frame": frame,
        "Time_seconds": time_sec,
        "Time_minutes": time_min,
        "OralNeuron1_Row1_dFF": n1,
        "OralNeuron2_Row2_dFF": n2,
        "OralNeuron3_Row3_dFF": n3,
        "OralNeurons123Mean_dFF": oral_mean,
        "PeduncleROI_dFF": ped,
        "StimStateProtocol": stim_state,
    })
    all_rows.append(df)

    meta_rows.append({
        "HydraID": hid,
        "OralRows1Based": "1,2,3",
        "PeduncleRow": "last row",
        "FramesUsed": n_use,
        "Duration_minutes": float(n_use / FPS / 60),
        "StimOnsetsCount": int(len(onset_frames)),
    })

    # visualization per Hydra: 4 panels
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False, gridspec_kw={"height_ratios": [1.1, 1.1, 1.1, 0.8]})

    # whole 30 min
    axes[0].plot(time_min, oral_mean, color="#1f77b4", lw=1.2, label="Oral neurons (rows 1,2,3) mean dF/F")
    axes[0].plot(time_min, ped, color="#d62728", lw=1.1, alpha=0.9, label="Peduncle ROI dF/F")
    axes[0].set_title(f"{hid}: Whole 30-min activity")
    axes[0].set_ylabel("dF/F")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    # spontaneous 20 min
    m20 = time_min <= SPONT_MIN
    axes[1].plot(time_min[m20], oral_mean[m20], color="#1f77b4", lw=1.2)
    axes[1].plot(time_min[m20], ped[m20], color="#d62728", lw=1.1, alpha=0.9)
    axes[1].set_title(f"{hid}: 20-min spontaneous activity (no stimulation)")
    axes[1].set_ylabel("dF/F")
    axes[1].grid(alpha=0.25)

    # stimulation 10 min
    m10 = (time_min >= SPONT_MIN) & (time_min <= SPONT_MIN + STIM_MIN)
    axes[2].plot(time_min[m10], oral_mean[m10], color="#1f77b4", lw=1.2)
    axes[2].plot(time_min[m10], ped[m10], color="#d62728", lw=1.1, alpha=0.9)
    axes[2].set_title(f"{hid}: 10-min stimulation activity")
    axes[2].set_ylabel("dF/F")
    axes[2].grid(alpha=0.25)

    # protocol at bottom: gray vertical lines for stimulus onsets
    axes[3].set_title(f"{hid}: Stimulation protocol (gray vertical lines)")
    axes[3].set_xlim(0, TOTAL_MIN)
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    axes[3].set_xlabel("Time (min)")
    for fr in onset_frames:
        axes[3].axvline(fr / FPS / 60.0, color="0.55", lw=0.8, alpha=0.8)
    axes[3].grid(alpha=0.15)

    fig.suptitle(f"{hid}: Oral (rows 1-3) mean dF/F and peduncle ROI dF/F", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_fig = os.path.join(OUT_DIR, f"Fig4_{hid}_Oral123_Peduncle_30min_20min_10min_protocol.png")
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)

# save dataset and metadata
full_df = pd.concat(all_rows, ignore_index=True)
meta_df = pd.DataFrame(meta_rows)

out_csv = os.path.join(OUT_DIR, "Fig4_Oral123_and_Peduncle_dFF_dataset.csv")
out_meta = os.path.join(OUT_DIR, "Fig4_Oral123_and_Peduncle_metadata.csv")
out_summary = os.path.join(OUT_DIR, "Fig4_Oral123_and_Peduncle_summary.txt")

full_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
meta_df.to_csv(out_meta, index=False, encoding="utf-8-sig")

with open(out_summary, "w", encoding="utf-8") as f:
    f.write("Fig4 oral dataset (fixed oral neurons rows 1,2,3) generated.\n")
    f.write("dF/F: cumulative-min baseline (F0 = minimum of prior timepoints).\n")
    f.write(f"Hydra IDs: {', '.join(HYDRA_IDS)}\n")
    f.write(f"Protocol for plotting: {SPONT_MIN} min off + {STIM_MIN} min repeated ({STIM_ON_SEC}s on/{STIM_OFF_SEC}s off).\n")
    f.write(f"FPS used: {FPS}\n")
    f.write(f"Rows in dataset: {len(full_df)}\n")

print('saved:', out_csv)
print('saved:', out_meta)
print('saved:', out_summary)
for hid in HYDRA_IDS:
    print('saved:', os.path.join(OUT_DIR, f"Fig4_{hid}_Oral123_Peduncle_30min_20min_10min_protocol.png"))
print('shape:', full_df.shape)
