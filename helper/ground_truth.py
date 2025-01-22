import numpy as np
import h5py

from helper.calculate_RR import fourier, peak_detection

'''
Change this code
'''


def ground_truth_rr(file_path, total_duration, fps, window_size=10, step_size=1):
    # Derived values
    samples_per_window = window_size * fps
    step_samples = step_size * fps

    # Load the data from the HDF5 file
    with h5py.File(file_path, "r") as hdf_file:
        respiration_signal = hdf_file['respiration'][:]

        # Adjust total_duration based on the actual data size
    total_samples = len(respiration_signal)
    total_duration = total_samples / fps  # Adjust total_duration to match data size

    # Validate data size
    if len(respiration_signal) != total_samples:
        raise ValueError("Mismatch between provided total duration and data size in the file.")

    # Sliding window calculation
    rr_values = []

    for start in range(0, total_samples - samples_per_window + 1, step_samples):
        end = start + samples_per_window
        window_signal = respiration_signal[start:end]
        rr = fourier(window_signal, fps)

        rr_values.append(rr)

    return rr_values
