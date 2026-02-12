import mne
import numpy as np


bdf_files = [
    "Lucas_FlexoExtension_Motor_Der2.bdf",
    "Lucas_FlexoExtension_Motor_Der3.bdf",
    "Lucas_FlexoExtension_Motor_Der4.bdf",
    "Lucas_FlexoExtension_Motor_Der5.bdf",
    "Lucas_FlexoExtension_Motor_Der6.bdf",
    "Lucas_FlexoExtension_Motor_Der7.bdf",
    "Lucas_FlexoExtension_Motor_Der8.bdf",
    "Lucas_FlexoExtension_Motor_Der9.bdf",
]

EVENT_ID = {
    "extension": 1,
    "cerrado": 2
}

TMIN = -1
TMAX = 0.5
WINDOW_LENGTH = 1.5


print("Cargando EEG...")
raws = [
    mne.io.read_raw_bdf(
        f,
        preload=True,
        stim_channel="Status"
    )
    for f in bdf_files
]

raw = mne.concatenate_raws(raws)
raw.pick_types(eeg=True)

print("Extrayendo eventos...")
events = mne.find_events(raw, stim_channel="Status")

print("Creando epochs...")
epochs = mne.Epochs(
    raw,
    events,
    event_id=EVENT_ID,
    tmin=TMIN,
    tmax=TMAX,
    baseline=None,
    preload=True
)

X = epochs.get_data()
y = epochs.events[:, -1]

print(f"Dataset: {X.shape[0]} trials, "
      f"{X.shape[1]} canales, "
      f"{X.shape[2]} muestras")