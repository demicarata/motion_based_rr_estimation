import numpy as np
from scipy.signal import hilbert

from helper.visualisation import plot_window


# How do I make these better


def cross_spectral_density_correlation(signal1, signal2):
    fft_signal1 = np.fft.fft(signal1)
    fft_signal2 = np.fft.fft(signal2)

    # Compute the Cross-Spectral Density (CSD)
    csd = np.conj(fft_signal1) * fft_signal2

    psd_signal1 = np.abs(fft_signal1) ** 2
    psd_signal2 = np.abs(fft_signal2) ** 2

    cross_spectral_density = np.abs(np.sum(csd)) / np.sqrt(np.sum(psd_signal1) * np.sum(psd_signal2))
    print(f"Cross Spectral Density: {cross_spectral_density}")

    return cross_spectral_density


def hilbert_correlation(signal1, signal2):
    analytic_signal1 = hilbert(signal1)
    analytic_signal2 = hilbert(signal2)

    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)

    phase_difference = phase1 - phase2

    mpc = np.abs(np.mean(np.exp(1j * phase_difference)))
    print(f"Mean Phase Coherence (MPC): {mpc:.3f}")

    return mpc
