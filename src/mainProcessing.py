import soundfile as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import kaiserord, lfilter, firwin, find_peaks, convolve, fftconvolve
from multiprocessing import Pool
from wAnalysis import W_analysis
import time
import customPlot

def load_audio(userInput):
    """
    Load 4 sinesweeps and returns the IR for FLU, FRD, BLD and BRU.
    Slices the results after the FFT and saves the .wav files in case
    the user needs them for further analisys.
    """    
    # Load LF, RF, LB and RB sinesweeps.
    FLU, sr = sf.read(userInput['FLU'][0])
    FRD, _ = sf.read(userInput['FRD'][0])
    BLD, _ = sf.read(userInput['BLD'][0])
    BRU, _ = sf.read(userInput['BRU'][0])
    inverse_filter, sr_filter = sf.read(userInput['InverseFilter'][0])

    assert sr == sr_filter, "Sampling rates do not match"

    if FLU.ndim > 1: FLU = FLU[:, 0]
    if FRD.ndim > 1: FRD = FRD[:, 0]
    if BLD.ndim > 1: BLD = BLD[:, 0]
    if BRU.ndim > 1: BRU = BRU[:, 0]
    if inverse_filter.ndim > 1: inverse_filter = inverse_filter[:, 0]

    # Sinesweep to IR.
    IRs = [fftconvolve(sig, inverse_filter, mode='full') for sig in [FLU, FRD, BLD, BRU]]

    # Get the peak of the IR, not the whole fftconvolve result.
    W = IRs[0] + IRs[1] + IRs[2] + IRs[3]
    max_idx = np.argmax(np.abs(W))

    # Slice: 1s before to 9s after the peak
    start = max(0, int(max_idx - sr * 1))
    end = int(max_idx + sr * 9) if (max_idx + sr * 9) < len(W) else len(W)
    t = (np.arange(start, end) - max_idx) / sr  # time = 0 at max W

    # Save sliced filtered files
    for name, data in zip(['FLU', 'FRD', 'BLD', 'BRU'], IRs):
        sf.write(f"{name}_IR.wav", data[start:end], sr)

    return (
        IRs[0][start:end],
        IRs[1][start:end],
        IRs[2][start:end],
        IRs[3][start:end],
        sr,
        t
    )

def convert_A_to_B(FLU, BLD, FRD, BRU, Fs):
    W = FLU + BLD + FRD + BRU
    X = FLU - BLD + FRD - BRU # FRONT
    Y = FLU + BLD - FRD - BRU # LEFT
    Z = FLU - BLD - FRD + BRU # UP

    # Save format B files.
    for name, signal in zip(["W_IR", "X_IR", "Y_IR", "Z_IR"], [W, X, Y, Z]):
        sf.write(f"{name}.wav", signal.astype(np.float32), Fs)

    return W, X, Y, Z

def compute_intensity(W, X, Y, Z, Fs, t, userInput):
    """
    Compute acoustic intensity from FOA channels.
    - 5 kHz low-pass filtering.
    - Smoothing with integration_time (s).
    - Returns recorte de 10 s alrededor del máximo.
    """    
    def filter_signals(Fs, W, X, Y, Z):
        Fc = 5000
        nyq_rate = Fs / 2.0
        width = 50.0 / Fs
        ripple_db = 60.0
        N, beta = kaiserord(ripple_db, width)
        taps = firwin(N, Fc / nyq_rate, window=('kaiser', beta))
        return (lfilter(taps, 1.0, sig) for sig in (W, X, Y, Z))
    integration_time = float(userInput["IntegrationTime_ms"].iloc[0]) / 1000  # ms
    W, X, Y, Z = filter_signals(Fs, W, X, Y, Z)
    # Get the Intensity.
    # Smooth, according to user pick.
    vent = int(np.round(integration_time * Fs))
    window = np.hamming(vent)
    window /= np.sum(window)
    Ix = np.convolve(W * X, window, mode='valid')
    Iy = np.convolve(W * Y, window, mode='valid')
    Iz = np.convolve(W * Z, window, mode='valid')

    I = np.sqrt(Ix**2 + Iy**2 + Iz**2)

    # Compute the angle of incidence.
    az = np.arctan2(Iy, Ix)
    el = np.arcsin(np.clip(Iz / (I + 1e-30), -1, 1))
    az_deg = np.rad2deg(az)
    el_deg = np.rad2deg(el)

    # Get maximum peak sample to slice the results.
    max_idx = np.argmax(I)
    start = max(0, max_idx - int(1 * Fs))
    end = min(len(I), max_idx + int(9 * Fs))

    I = I[start:end]
    az_deg = az_deg[start:end]
    el_deg = el_deg[start:end]
    t_new = (np.arange(start, end) - max_idx) / Fs

    return I, az_deg, el_deg, t_new

def find_peaks_info(intensity, azimuth, elevation, time):
    """
    Find peaks in intensity signal starting from the maximum amplitude.
    """

    peaks_idx = find_peaks(intensity)[0]
    I_max_amp = np.argmax(intensity)

    # Maximum value is the first peak. Keeps the following values.
    I_max_peak = np.argmin(np.abs(peaks_idx - I_max_amp))
    peaks_idx = peaks_idx[I_max_peak:]

    # Extract peaks data
    intensity_peaks = intensity[peaks_idx]
    azimuth_peaks = azimuth[peaks_idx]
    elevation_peaks = elevation[peaks_idx]
    time_peaks = time[peaks_idx]

    # Compute relative time (ms) zeroed at first peak time
    time_rel = (time_peaks - time_peaks[0]) * 1e3

    return intensity_peaks, azimuth_peaks, elevation_peaks, time_rel

def export_peaks(I, az, el, t, threshold_dB, filename="intensity_table.csv"):
    I_max = np.max(I) + 1e-30
    I_dB = 20 * np.log10(I / I_max)

    mask = I_dB >= threshold_dB

    I_dB_thresh = I_dB[mask]
    az_thresh = az[mask]
    el_thresh = el[mask]
    time_thresh = t[mask] * 1e3  # ms

    azimuth_deg = np.rad2deg(az_thresh)
    elevation_deg = np.rad2deg(el_thresh)

    data = np.column_stack((time_thresh, I_dB_thresh, azimuth_deg, elevation_deg))
    df = pd.DataFrame(data, columns=['Time [ms]', 'Magnitude [dB]', 'Azimuth [°]', 'Elevation [°]'])

    # Exportar a CSV para abrir directo en Excel
    df.to_csv(filename, index=False, sep=';')

def main_processing(userInput):
    # Load FLU, FRD, BLD and BRU sinesweeps, returns IRs.
    FLU, FRD, BLD, BRU, sr, t = load_audio(userInput)

    # Ambisonics A to B conversion.
    W, X, Y, Z = convert_A_to_B(FLU, BLD, FRD, BRU, sr)

    # Compute the intensity and angle of incidence for the acoustic sound.
    I, az, el, t = compute_intensity(W, X, Y, Z, sr, t, userInput)

    # Points of interest for the hedgehog plot.
    I_peaks, az_peaks, el_peaks, time_peaks = find_peaks_info(I, az, el, t)

    # Plot
    customPlot.plot_intensity_3d(I_peaks, az_peaks, el_peaks, time_peaks, t, userInput)

    # Output data table and process W.
    threshold_dB = int(userInput["Threshold_dB"].iloc[0])
    export_peaks(I, az, el, t, threshold_dB)
    band_filter = userInput["Band_Filter"].iloc[0]
    W_analysis(W, sr, band_filter)
    
    
    
    
    
    
    