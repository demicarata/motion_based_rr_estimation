import scipy.signal as sig
from scipy.signal import filtfilt, butter, savgol_filter, find_peaks
from sklearn.decomposition import PCA
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter1d


def bandpass_filter(signal, fs, lowcut, highcut, order=8):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y


def moving_average_filter(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, 'same')


def median_filter(signal, window_size=5):
    return np.array([np.median(signal[max(0, i - window_size // 2): i + window_size // 2 + 1]) for i in range(len(signal))])


def exponential_moving_average(signal, alpha=0.2):
    ema_signal = np.zeros_like(signal)
    ema_signal[0] = signal[0]
    for i in range(1, len(signal)):
        ema_signal[i] = alpha * signal[i] + (1 - alpha) * ema_signal[i -1]
    return ema_signal


def apply_pca(data, n_components=1):
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(data)  # Reduce dimensions
    reconstructed_signal = pca.inverse_transform(pca_transformed)  # Reconstruct signal
    return reconstructed_signal[:, 0], pca.explained_variance_ratio_


def wavelet_denoising(signal, wavelet='db4', level=4, threshold=None):
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    if threshold is None:
        threshold = np.sqrt(2 * np.log(len(signal)))

    coeffs_thresholded = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)
    return denoised_signal


def gaussian_filter(signal):
    return gaussian_filter1d(signal, sigma=1)


def savgol(signal):
    return savgol_filter(signal, window_length=5, polyorder=3)


def magnitude_threshold_filter(signal, threshold_factor=1.0):
    # Compute the magnitude of the signal (absolute values)
    magnitudes = np.abs(signal)

    # Calculate the average magnitude of the signal
    avg_magnitude = np.mean(magnitudes)

    # Set a threshold based on the average magnitude
    threshold = avg_magnitude * threshold_factor

    # Identify peaks (values larger than their neighbors)
    peaks, _ = sig.find_peaks(signal)  # Find peaks in the signal

    # Suppress peaks below the threshold
    filtered_signal = signal.copy()
    for peak in peaks:
        if np.abs(signal[peak]) < threshold:
            filtered_signal[peak] = 0  # Remove the peak by setting it to zero

    return filtered_signal, avg_magnitude, threshold
