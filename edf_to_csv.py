import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# === Config ===
datapath = '/Volumes/FF952/datasets/Sleep_stage_EEG'
output_csv = "sleep_all_patients.csv"
visualize = False
fallback_channels = ['Fpz-Cz', 'Pz-Oz', 'EEG Fpz-Cz', 'EEG Pz-Oz']
label_map = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

# === Step 1: Find file pairs ===
all_files = sorted(os.listdir(datapath))
psg_files = [f for f in all_files if "-PSG.edf" in f and "._" not in f]
hypno_files = [f for f in all_files if "-Hypnogram.edf" in f and "._" not in f]

# Build (psg, hypnogram) pairs by prefix match
subject_pairs = []
for psg in psg_files:
    prefix = psg.split("-")[0][:-1]
    hypnos = [h for h in hypno_files if prefix in h]
    if hypnos:
        subject_pairs.append((os.path.join(datapath, psg), os.path.join(datapath, hypnos[0])))

print(f"Found {len(subject_pairs)} subject pairs.")

# === Step 2: Determine common channel across all subjects ===
channel_sets = []
for signal_file, _ in subject_pairs:
    raw = mne.io.read_raw_edf(signal_file, preload=False, verbose=False)
    channel_sets.append(set(raw.info['ch_names']))

common_channels = set.intersection(*channel_sets)
chosen_channel = None
for ch in fallback_channels:
    if ch in common_channels:
        chosen_channel = ch
        break

if not chosen_channel:
    raise ValueError("No common fallback EEG channel found across subjects.")

print(f"Using EEG channel: {chosen_channel}")

# === Step 3: Process each subject ===
all_X, all_y = [], []

for i, (signal_file, hypnogram_file) in enumerate(subject_pairs):
    print(f"\nProcessing {os.path.basename(signal_file)}...")

    raw = mne.io.read_raw_edf(signal_file, preload=True, verbose=False)
    raw.pick_channels([chosen_channel])
    raw.filter(0.5, 30., fir_design='firwin')

    annotations = mne.read_annotations(hypnogram_file)
    raw.set_annotations(annotations)

    events, _ = mne.events_from_annotations(raw, event_id=label_map, verbose=False)

    if not events.any():
        print("  No events found – skipping.")
        continue

    epochs = mne.Epochs(raw, events, event_id=None, tmin=0.0, tmax=30.0, baseline=None, detrend=1, verbose=False)
    X = epochs.get_data()[:, 0, :]  # (n_epochs, n_times)
    y = epochs.events[:, -1]

    print(f"  → {len(X)} epochs")

    all_X.append(X)
    all_y.append(y)

    # Visualize only first file
    if visualize and i == 0:
        raw.plot(scalings='auto', duration=120, title=f"EEG Signal: {chosen_channel}", show=True)

        stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
        label_counts = pd.Series(y).value_counts().sort_index()
        label_names = [stage_names[i] for i in label_counts.index]

        plt.figure(figsize=(6, 4))
        sns.barplot(x=label_names, y=label_counts.values)
        plt.title("Sleep Stage Distribution")
        plt.ylabel("Count")
        plt.xlabel("Stage")
        plt.show()

# === Step 4: Combine and Save ===
X_all = np.concatenate(all_X)
y_all = np.concatenate(all_y)

df = pd.DataFrame(X_all)
df['stage'] = y_all
df.to_csv(output_csv, index=False)

print(f"\n Saved combined dataset: {output_csv}")
print(f"Total samples: {len(df)}")
