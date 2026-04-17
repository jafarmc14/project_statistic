import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def bandpass(signal, fs=1000, low=20, high=450, order=4):
    if signal.ndim == 1:
        b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
        return filtfilt(b, a, signal)
    # apply per-channel
    out = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
        out[:, i] = filtfilt(b, a, signal[:, i])
    return out

def notch(signal, fs=1000, freq=50.0, Q=30.0):
    if signal.ndim == 1:
        b, a = iirnotch(freq/(fs/2), Q)
        return filtfilt(b, a, signal)
    out = np.zeros_like(signal)
    b, a = iirnotch(freq/(fs/2), Q)
    for i in range(signal.shape[1]):
        out[:, i] = filtfilt(b, a, signal[:, i])
    return out

def preprocess(x, fs=1000, rectify=True, normalize=True):
    """Preprocess EMG: bandpass -> notch 50 Hz -> (abs) -> (normalize). Accepts 1D or 2D (time x channels)."""
    y = bandpass(x, fs=fs)
    y = notch(y, fs=fs, freq=50.0, Q=30.0)
    if rectify:
        y = np.abs(y)
    if normalize:
        # per-channel max normalization
        if y.ndim == 1:
            m = np.max(np.abs(y)) + 1e-8
            y = y / m
        else:
            m = np.max(np.abs(y), axis=0, keepdims=True) + 1e-8
            y = y / m
    return y
