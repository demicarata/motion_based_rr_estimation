import numpy as np
import h5py

from helper.calculate_RR import fourier, peak_detection


def ground_truth_rr(file_path, total_duration, fps, window_size=10, step_size=1):
    # Derived values
    samples_per_window = window_size * fps
    step_samples = step_size * fps
    total_samples = total_duration * fps
    min_rr = 10
    max_rr = 48

    # Load the data from the HDF5 file
    with h5py.File(file_path, "r") as hdf_file:
        respiration_signal = hdf_file['respiration'][:]

        # Adjust total_duration based on the actual data size
    total_samples = len(respiration_signal)
    total_duration = total_samples / fps  # Adjust total_duration to match data size

    # Validate data size
    if len(respiration_signal) != total_samples:
        raise ValueError("Mismatch between provided total duration and data size in the file.")

    # Function to calculate RR in a window using FFT
    def compute_rr_fft(signal, sampling_rate):
        # Apply FFT
        fft_result = np.fft.fft(signal)
        fft_freqs = np.fft.fftfreq(len(signal), d=1 / sampling_rate)

        # Select positive frequencies
        positive_freqs = fft_freqs[fft_freqs > 0]
        positive_fft = np.abs(fft_result[fft_freqs > 0])

        # Convert respiratory rate limits to Hz
        min_freq = min_rr / 60
        max_freq = max_rr / 60

        # Apply bandpass filter
        valid_indices = (positive_freqs >= min_freq) & (positive_freqs <= max_freq)
        filtered_freqs = positive_freqs[valid_indices]
        filtered_fft = positive_fft[valid_indices]

        if len(filtered_fft) == 0:
            # No valid frequencies in the desired range
            return None

        # Find the dominant frequency within the band
        dominant_freq = filtered_freqs[np.argmax(filtered_fft)]

        # Convert frequency to breaths per minute
        rr_bpm = dominant_freq * 60
        return rr_bpm


    # Sliding window calculation
    rr_values = []

    for start in range(0, total_samples - samples_per_window + 1, step_samples):
        end = start + samples_per_window
        window_signal = respiration_signal[start:end]
        rr = fourier(window_signal, fps)
        rr_values.append(rr)

    return rr_values
