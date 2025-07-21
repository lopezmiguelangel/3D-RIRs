import numpy as np
from scipy.signal import butter, sosfilt, hilbert
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl

def trim(W, Fs):
    # Avoid extremely large audio files.
    idx_max = np.argmax(np.abs(W))
    max_dB = 20 * np.log10(np.abs(W[idx_max]) + 1e-30)
    levels_dB = 20 * np.log10(np.abs(W[:idx_max+1]) + 1e-30)
    threshold = max_dB - 20
    idx_20dB = np.where(levels_dB < threshold)[0]
    start_idx = idx_20dB[-1] if len(idx_20dB) > 0 else 0
    end_idx = min(start_idx + int(Fs*10), len(W)) 
    return W[start_idx:end_idx]

def band_filter(signal, Fs, band="octave"):
    # Generates the filters according to the user input.
    fc_octave = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    fc_third = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000]
    fc_list = fc_octave if band == "octave" else fc_third

    G = 10**(3/10)
    if band == "third":
        fmin = G**(-1/6)
        fmax = G**(1/6)
    else:
        fmin = G**(-1/2)
        fmax = G**(1/2)

    signal = np.flip(signal)
    filtered = []

    for fc in fc_list:
        low = (fc * fmin) / (Fs / 2)
        high = (fc * fmax) / (Fs / 2)
        if high >= 0.99:
            high = 0.99
        sos = butter(N=2, Wn=[low, high], btype='bandpass', output='sos')
        f = sosfilt(sos, signal)
        f = np.flip(f)
        f = f[:int(len(f) * 0.95)]
        filtered.append(f)

    min_len = min(len(signal), *(len(f) for f in filtered))
    signal = signal[:min_len]
    filtered = [f[:min_len] for f in filtered]

    all_signals = [signal] + filtered
    return np.array(all_signals), fc_list

def find_noise_floor(signal, Fs, noise_floor = None):
    # Improves results for the Schroeder smoothing.
    minimum = np.max(signal)
    new_min = len(signal)
    for j in np.arange(0.01, 1.01, 0.01):
        step_min = int(Fs * j)
        flag = False
        try:
            for i in range(np.argmax(signal), len(signal) - step_min, step_min):
                mean_val = np.mean(signal[i:i + step_min])
                if minimum > mean_val and not flag:
                    new_min = i
                    minimum = mean_val
                if minimum < mean_val:
                    if noise_floor is not None and minimum < (noise_floor + 3):
                        flag = True
                        break
        except Exception:
            return None
    return new_min

def process_band(W, Fs):
    # Smooth W and then calculate RT20, RT30, RT60, C59, C80, cT, EDT.
    
    # First apply dB scala.
    smoothed_1 = 10 * np.log10((W ** 2) / np.sum(W ** 2) + 1e-30)
    smoothed_1 = smoothed_1 - np.max(smoothed_1)

    # Keep signal, remove noise floor.
    noise_floor = np.mean(smoothed_1[len(smoothed_1)//2:])
    new_min = find_noise_floor(smoothed_1, Fs, noise_floor)

    # Apply Schroeder.
    smoothed_2 = W[:new_min]
    energy = np.cumsum(smoothed_2[::-1] ** 2)
    energy /= energy[-1]
    smoothed_2 = 10 * np.log10(energy + 1e-30)[::-1]

    max_idx = np.argmax(smoothed_2)
    smoothed_2 = smoothed_2[max_idx:] - np.max(smoothed_2[max_idx:])

    def find_first_leq(arr, val):
        idxs = np.where(arr <= val)[0]
        return idxs[0] if len(idxs) > 0 else None

    # Parameters calculations.
    EDT = (find_first_leq(smoothed_2, -10) or 0) / Fs * 6
    RT20 = ((find_first_leq(smoothed_2, -25) or 0) - (find_first_leq(smoothed_2, -5) or 0)) * 3 / Fs
    RT30 = ((find_first_leq(smoothed_2, -35) or 0) - (find_first_leq(smoothed_2, -5) or 0)) * 2 / Fs

    # Aux signal to obtain C and D parameters.
    clar_idx_ini = np.argmax(smoothed_1)
    if noise_floor < -35:
        clar_idx_end = int(Fs * (RT20 + EDT + RT30) / 3 + clar_idx_ini)
    elif noise_floor < -25:
        clar_idx_end = int((RT20 + EDT) / 2 * Fs + clar_idx_ini)
    else:
        clar_idx_end = int(EDT * Fs + clar_idx_ini)

    clar_signal = W[clar_idx_ini:clar_idx_end] ** 2

    C50 = 10 * np.log10(
        np.sum(clar_signal[:int(0.050 * Fs)]) /
        (np.sum(clar_signal[int(0.050 * Fs):]) + 1e-30)
    )
    C80 = 10 * np.log10(
        np.sum(clar_signal[:int(0.080 * Fs)]) /
        (np.sum(clar_signal[int(0.080 * Fs):]) + 1e-30)
    )
    D50 = np.sum(clar_signal[:int(0.050 * Fs)]) / np.sum(clar_signal) * 100
    D80 = np.sum(clar_signal[:int(0.080 * Fs)]) / np.sum(clar_signal) * 100

    indices = np.arange(1, len(clar_signal) + 1)
    cT = np.round(np.sum(indices * clar_signal) / np.sum(clar_signal))

    EDTtsgnl1 = smoothed_2[int(cT):]
    idx_EDTt = np.where(EDTtsgnl1 <= -10)[0]
    EDTt = ((idx_EDTt[0] - np.argmax(EDTtsgnl1)) / Fs * 6) if len(idx_EDTt) > 0 else np.nan

    SNR = -noise_floor
    cT_ms = cT / Fs * 1000

    return {
        "C50": C50, "C80": C80, "EDT": EDT, "RT20": RT20, "RT30": RT30,
        "SNR": SNR, "D50": D50, "D80": D80, "cT_ms": cT_ms, "EDTt": EDTt
    }

def exportW(df_results, W, Fs):
    fig, ax = plt.subplots(figsize=(10, len(df_results)*0.4 + 1))
    ax.axis('off')
    table = ax.table(
        cellText=df_results.round(2).values,
        colLabels=df_results.columns,
        rowLabels=df_results.index,
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title("Acoustics", fontsize=14)
    plt.tight_layout()
    plt.savefig("Acoustics.png", dpi=600)
    plt.show()
    
    smoothed_1 = 10 * np.log10((W ** 2) / np.sum(W ** 2) + 1e-30)
    smoothed_1 = smoothed_1 - np.max(smoothed_1)

    noise_floor = np.mean(smoothed_1[len(smoothed_1)//2:])
    new_min = find_noise_floor(smoothed_1, Fs, noise_floor)

    smoothed_2 = W[:new_min]
    energy = np.cumsum(smoothed_2[::-1] ** 2)
    energy /= energy[-1]
    smoothed_2 = 10 * np.log10(energy + 1e-30)[::-1]

    max_idx = np.argmax(smoothed_2)
    smoothed_2 = smoothed_2[max_idx:] - np.max(smoothed_2[max_idx:])

    min_len = min(len(smoothed_1), len(smoothed_2))

    # Plot downsampled signal. Improves speed.
    step = 20
    start_idx = int(0.1*Fs)  # Skip first samples
    min_len = min(len(smoothed_1), len(smoothed_2))
    smoothed_1 = smoothed_1[start_idx:min_len][::step]
    smoothed_2 = smoothed_2[start_idx:min_len][::step]
    times = np.arange(start_idx, min_len, step) / Fs
    offset = np.percentile(smoothed_2 - smoothed_1, 0.5)
    smoothed_1 = smoothed_1 + offset

    plt.figure(figsize=(10, 5))
    plt.plot(times, smoothed_1, label="Accumulated energy (dB)", color="gray")
    plt.plot(times, smoothed_2, label="Accumulated energy (Schroeder)", color="blue")
    plt.axhline(-5, color='gray', linestyle='--', label="-5 dB")
    plt.axhline(-10, color='red', linestyle='--', label="-10 dB (EDT)")
    plt.axhline(-25, color='blue', linestyle='--', label="-25 dB (RT20)")
    plt.axhline(-35, color='green', linestyle='--', label="-35 dB (RT30)")
    plt.xlabel("Time (s)")
    plt.ylabel("Level (dB)")
    plt.title("Accumulated Energy Curve and Decay Levels")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(-80, 1)

    # Export png.
    plt.savefig("W_Smoothed_Curves.png", dpi=600)
    plt.show()

    # Export as cvs and xlsx
    df_results.to_excel("Acoustics.xlsx", sheet_name="Acoustic_Results")
    df_results.to_csv("Acoustics.csv")

def W_analysis(W, Fs, band):
    W = trim(W, Fs)
    
    # Global
    results = []
    global_result = process_band(W, Fs)
    global_result["Band (Hz)"] = "Global"
    results.append(global_result)

    # By band.
    filtered_bands, band_names = band_filter(W, Fs, band)
    for i, filtered_signal in enumerate(filtered_bands[1:]):
        result = process_band(filtered_signal, Fs)
        result["Band (Hz)"] = band_names[i]
        results.append(result)

    df_results = pd.DataFrame(results).set_index("Band (Hz)")

    exportW(df_results, W, Fs)
