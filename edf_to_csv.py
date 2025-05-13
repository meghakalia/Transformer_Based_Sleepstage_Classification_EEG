import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# === Config ===
datapath = '/Volumes/FF952/datasets/Sleep_stage_EEG'
signal_file = os.path.join(datapath, "SC4001E0-PSG.edf")
hypnogram_file = os.path.join(datapath, "SC4001EC-Hypnogram.edf")
output_csv = "sleep_patient1.csv"

# === Choose a valid EEG channel from fallback list ===
# fallback_channels = ['Fpz-Cz', 'Pz-Oz', 'EEG Fpz-Cz', 'EEG Pz-Oz']

target_channel = "Fpz-Cz"  # or try "Pz-Oz" if missing
visualize = True

# === Load and Preprocess ===
print("Reading signal...")
raw = mne.io.read_raw_edf(signal_file, preload=True, verbose=False)

fallback_channels = ['Fpz-Cz', 'Pz-Oz', 'EEG Fpz-Cz', 'EEG Pz-Oz']

available_channels = raw.info['ch_names']
print("Available channels:", available_channels)

chosen_channel = None
for ch in fallback_channels:
    if ch in available_channels:
        chosen_channel = ch
        break

if chosen_channel is None:
    raise ValueError("None of the fallback EEG channels found in this EDF file.")

print(f"Using EEG channel: {chosen_channel}")
raw.pick_channels([chosen_channel])


# raw.pick_channels([target_channel])
raw.filter(0.5, 30., fir_design='firwin')  # Bandpass filter

print("Reading annotations...")
annotations = mne.read_annotations(hypnogram_file)
raw.set_annotations(annotations)

# === Visualize raw EEG ===
if visualize:
    raw.plot(scalings='auto', duration=120, title="EEG Signal", show=True)

# === Label Mapping ===
label_map = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # Merge N3
    "Sleep stage R": 4,
}
events, _ = mne.events_from_annotations(raw, event_id=label_map, verbose=False)

# === Epoch into 30s ===
epochs = mne.Epochs(
    raw, events, event_id=None, tmin=0.0, tmax=30.0,
    baseline=None, detrend=1, verbose=False
)
X = epochs.get_data()[:, 0, :]  # shape: (n_epochs, n_times)
y = epochs.events[:, -1]

print(f"Extracted {len(X)} epochs with {X.shape[1]} samples each.")

# === Visualize label distribution ===
if visualize:
    stage_names = ['W', 'N1', 'N2', 'N3', 'REM']
    label_counts = pd.Series(y).value_counts().sort_index()
    label_names = [stage_names[i] for i in label_counts.index]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=label_names, y=label_counts.values)
    plt.title("Sleep Stage Distribution")
    plt.ylabel("Count")
    plt.xlabel("Stage")
    plt.show()

# === Save to CSV ===
df = pd.DataFrame(X)
df["stage"] = y
df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")


