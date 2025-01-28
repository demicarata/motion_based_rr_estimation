import numpy as np


def fourier(signal, fps, lowcut=0.3, highcut=0.8):
    # Apply FFT
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1 / fps)
    fft_magnitude = np.abs(np.fft.rfft(signal))

    # Filter frequencies within respiratory range
    valid_indices = (freqs >= lowcut) & (freqs <= highcut)
    valid_freqs = freqs[valid_indices]
    valid_magnitudes = fft_magnitude[valid_indices]

    if len(valid_freqs) == 0:
        return 0

    # Identify dominant frequency
    dominant_freq = valid_freqs[np.argmax(valid_magnitudes)]
    respiratory_rate_bpm = dominant_freq * 60
    return respiratory_rate_bpm


def peak_detection(signal, fps, min_peak_distance_time=0.5):
    min_peak_distance = int(fps * min_peak_distance_time)

    mean_signal = np.mean(signal)

    # Find indices of peaks
    peaks = np.where((signal[1:-1] > signal[:-2]) &
                     (signal[1:-1] > signal[2:]) &
                     (signal[1:-1] > mean_signal))[0] + 1

    # Filter peaks based on minimum peak distance
    filtered_peaks = []
    last_peak = -min_peak_distance
    for peak in peaks:
        if peak - last_peak >= min_peak_distance:
            filtered_peaks.append(peak)
            last_peak = peak

    # Calculate the respiratory rate
    breaths = len(filtered_peaks)
    duration_seconds = len(signal) / fps
    respiratory_rate = (breaths / duration_seconds) * 60

    return respiratory_rate
