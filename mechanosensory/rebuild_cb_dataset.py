import os
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Paths
OUT_DIR = r"C:\Users\ASUS\Documents\Hydra\Hydra_download_videos\mechanosensory"
IN_MAT = os.path.join(OUT_DIR, "elife-64108-fig2-data1-v2.mat")

# Output files (overwrite old results)
OUT_DATASET = os.path.join(OUT_DIR, "CB_dFF_31s_aligned_dataset.csv")
OUT_SEGMENTS = os.path.join(OUT_DIR, "CB_dFF_31s_aligned_segments.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "CB_dFF_31s_aligned_summary.txt")
OUT_FIG_4 = os.path.join(OUT_DIR, "CB_dFF_4segment_comparison.png")
OUT_FIG_SINGLE = [os.path.join(OUT_DIR, f"CB_dFF_segment_example_{i}.png") for i in range(1, 5)]

# Parameters
fs = 16.67
segment_sec = 31.0
pre_sec = 1.0  # stimulus onset aligned at t=1s
segment_frames = int(round(segment_sec * fs))
pre_frames = int(round(pre_sec * fs))

# Optional smoothing after dF/F (set to False to disable)
apply_smoothing = True
sg_order = 3
sg_frame = 11


def parse_pressure(cond: str):
    if cond.lower() == "nostim":
        return 0
    m = re.search(r"(\d+)", cond)
    return int(m.group(1)) if m else np.nan


def compute_dff_from_cummin(raw: np.ndarray) -> np.ndarray:
    # Literature definition: F0 is minimum fluorescence from prior timepoints.
    f0 = np.minimum.accumulate(raw)
    f0_safe = np.maximum(f0, np.finfo(np.float64).eps)
    dff = (raw - f0_safe) / f0_safe
    return dff


def maybe_smooth(y: np.ndarray) -> np.ndarray:
    if not apply_smoothing:
        return y
    n = len(y)
    if n < (sg_order + 2):
        return y
    frame = sg_frame
    if frame <= sg_order:
        frame = sg_order + 2
    if frame % 2 == 0:
        frame += 1
    if frame > n:
        frame = n if n % 2 == 1 else n - 1
    return savgol_filter(y, frame, sg_order, mode="interp")


if not os.path.exists(IN_MAT):
    raise FileNotFoundError(f"Cannot find MAT file: {IN_MAT}")

mat = loadmat(IN_MAT, squeeze_me=True, struct_as_record=False)
hydra = np.ravel(mat["HydraData"])

all_rows = []
seg_rows = []
segment_id = 0

for trial_idx, tr in enumerate(hydra, start=1):
    raw = np.asarray(tr.RawFluorescenceFoot, dtype=np.float64).ravel()
    stim = (np.asarray(tr.StimulationTrace, dtype=np.float64).ravel() > 0.5).astype(np.int8)

    cond = str(tr.StimCondition)
    psi = parse_pressure(cond)
    response_prob = float(getattr(tr, "ResponseProbability", np.nan))

    dff = compute_dff_from_cummin(raw)
    dff = maybe_smooth(dff)

    n = len(raw)

    # Stim segments in original order
    onsets = np.where(np.diff(stim.astype(int), prepend=0) == 1)[0]
    stim_segments = []
    for onset in onsets:
        start = onset - pre_frames
        end = start + segment_frames
        if start < 0 or end > n:
            continue
        stim_segments.append((start, end, onset, "stim"))

    # No-stim segments: find fully zero windows, keep non-overlap, match count
    zero = (stim == 0)
    conv = np.convolve(zero.astype(np.int32), np.ones(segment_frames, dtype=np.int32), mode="valid")
    candidates = np.where(conv == segment_frames)[0]

    no_stim_segments = []
    need = len(stim_segments)
    last_end = -1
    for s in candidates:
        e = int(s + segment_frames)
        if s >= last_end:
            virtual_onset = int(s + pre_frames)
            no_stim_segments.append((int(s), e, virtual_onset, "no_stim"))
            last_end = e
            if len(no_stim_segments) >= need:
                break

    # Merge and sort by original timeline
    trial_segments = stim_segments + no_stim_segments
    trial_segments.sort(key=lambda x: x[0])

    for order_in_trial, (start, end, onset_like, seg_type) in enumerate(trial_segments, start=1):
        segment_id += 1
        seg_dff = dff[start:end]
        seg_stim = stim[start:end]
        t = np.arange(end - start, dtype=np.float64) / fs
        fidx = np.arange(1, end - start + 1, dtype=np.int32)

        all_rows.append(pd.DataFrame({
            "SegmentID": segment_id,
            "Trial": trial_idx,
            "SegmentOrderInTrial": order_in_trial,
            "StimCondition": cond,
            "StimPressurePsi": psi,
            "SegmentType": seg_type,
            "ResponseProbability": response_prob,
            "SourceOnsetFrame": int(onset_like + 1),
            "FrameInSegment": fidx,
            "Time_seconds": t,
            "dFF_CB": seg_dff,
            "StimState": seg_stim
        }))

        seg_rows.append({
            "SegmentID": segment_id,
            "Trial": trial_idx,
            "SegmentOrderInTrial": order_in_trial,
            "StimCondition": cond,
            "StimPressurePsi": psi,
            "SegmentType": seg_type,
            "ResponseProbability": response_prob,
            "SourceOnsetFrame": int(onset_like + 1),
            "StartFrame": int(start + 1),
            "EndFrame": int(end),
            "Duration_seconds": float((end - start) / fs)
        })

full_df = pd.concat(all_rows, ignore_index=True)
seg_df = pd.DataFrame(seg_rows)

# stable sort in original order
full_df = full_df.sort_values(["Trial", "SegmentOrderInTrial", "FrameInSegment"], kind="stable").reset_index(drop=True)
seg_df = seg_df.sort_values(["Trial", "SegmentOrderInTrial"], kind="stable").reset_index(drop=True)

# overwrite old results
full_df.to_csv(OUT_DATASET, index=False, encoding="utf-8-sig")
seg_df.to_csv(OUT_SEGMENTS, index=False, encoding="utf-8-sig")

# Select four example segments: first 2 stim + first 2 no_stim
stim_ids = seg_df.loc[(seg_df["SegmentType"] == "stim") & (seg_df["StimPressurePsi"] > 0), "SegmentID"].head(2).tolist()
nostim_ids = seg_df.loc[(seg_df["SegmentType"] == "no_stim") & (seg_df["StimPressurePsi"] == 0), "SegmentID"].head(2).tolist()
example_ids = stim_ids + nostim_ids

example_titles = ["Stim A", "Stim B", "NoStim A", "NoStim B"]

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, sid, title in zip(axes.ravel(), example_ids, example_titles):
    sdf = full_df[full_df["SegmentID"] == sid]
    t = sdf["Time_seconds"].to_numpy()
    y = sdf["dFF_CB"].to_numpy()
    st = sdf["StimState"].to_numpy()

    ax.plot(t, y, color="tab:blue", lw=1.2, label="dF/F (CB)")

    on = np.where(st > 0)[0]
    if len(on) > 0:
        runs = np.split(on, np.where(np.diff(on) != 1)[0] + 1)
        for r in runs:
            ax.axvspan(t[r[0]], t[r[-1]], color="orange", alpha=0.25)

    ax.axvline(1.0, color="red", linestyle="--", lw=1, label="Aligned onset @1s")
    meta = seg_df.loc[seg_df["SegmentID"] == sid].iloc[0]
    ax.set_title(f"{title} | ID={sid} | Trial={int(meta['Trial'])} | {meta['StimCondition']}")
    ax.set_ylabel("dF/F")
    ax.grid(alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel("Time (s)")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
fig.suptitle("CB dF/F (31s segments, onset aligned at 1s)")
fig.tight_layout(rect=[0, 0, 0.96, 0.95])
fig.savefig(OUT_FIG_4, dpi=220)
plt.close(fig)

# save four single plots
for i, sid in enumerate(example_ids, start=1):
    sdf = full_df[full_df["SegmentID"] == sid]
    t = sdf["Time_seconds"].to_numpy()
    y = sdf["dFF_CB"].to_numpy()
    st = sdf["StimState"].to_numpy()

    plt.figure(figsize=(8, 3.2))
    plt.plot(t, y, color="tab:blue", lw=1.2)

    on = np.where(st > 0)[0]
    if len(on) > 0:
        runs = np.split(on, np.where(np.diff(on) != 1)[0] + 1)
        for r in runs:
            plt.axvspan(t[r[0]], t[r[-1]], color="orange", alpha=0.25)

    plt.axvline(1.0, color="red", linestyle="--", lw=1)
    plt.xlabel("Time (s)")
    plt.ylabel("dF/F (CB)")
    plt.title(f"SegmentID {sid}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG_SINGLE[i - 1], dpi=220)
    plt.close()

with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
    f.write("Rebuilt dataset (overwritten old results).\n")
    f.write("dF/F definition: F0 = cumulative minimum of prior timepoints (literature).\n")
    f.write(f"Source MAT: {IN_MAT}\n")
    f.write(f"Trials: {len(hydra)}\n")
    f.write(f"Segments: {len(seg_df)}\n")
    f.write(f"Rows: {len(full_df)}\n")
    f.write(f"fs: {fs}\n")
    f.write(f"segment_sec: {segment_sec}, pre_sec: {pre_sec}\n")
    f.write(f"smoothing: {apply_smoothing}, sg_order: {sg_order}, sg_frame: {sg_frame}\n")
    f.write(f"Example segment IDs: {example_ids}\n")

print("Saved script outputs:")
print(OUT_DATASET)
print(OUT_SEGMENTS)
print(OUT_SUMMARY)
print(OUT_FIG_4)
for p in OUT_FIG_SINGLE:
    print(p)
print("Example IDs:", example_ids)

