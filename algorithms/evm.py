import cv2
import numpy as np
import time
import psutil

from helper.filters import bandpass_filter, moving_average_filter, exponential_moving_average
from helper.calculate_RR import fourier
from helper.visualisation import plot_window
from helper.window_correlation import hilbert_correlation, cross_spectral_density_correlation


def build_gaussian_pyramid(frame, levels=3):
    pyramid = [frame]
    for _ in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid


def reconstruct_laplacian_pyramid(pyramid):
    frame = pyramid[-1]
    for prev in reversed(pyramid[:-1]):
        frame = cv2.pyrUp(frame, dstsize=(prev.shape[1], prev.shape[0]))
        frame = cv2.add(frame, prev)
    return frame


def amplify_motion(video_frames, fps, alpha=100, low_freq=0.3, high_freq=0.8, levels=3):
    # Decompose frames into pyramids
    pyramids = [build_gaussian_pyramid(frame, levels) for frame in video_frames]
    amplified_pyramids = []

    # Apply temporal filtering and amplify motion
    for level in range(levels):
        print("Level", level)
        level_frames = np.array([p[level] for p in pyramids])
        filtered = bandpass_filter(level_frames, fps, low_freq, high_freq, 3)
        amplified_frames = level_frames + alpha * filtered
        amplified_pyramids.append(amplified_frames)

    # Reconstruct video frames
    amplified_frames = []
    for i in range(len(video_frames)):
        pyramid = [amplified_pyramids[level][i] for level in range(levels)]
        amplified_frames.append(reconstruct_laplacian_pyramid(pyramid))
    return amplified_frames


def extract_motion_signal(amplified_frames):
    signal = []
    for i in range(1, len(amplified_frames)):
        diff = amplified_frames[i] - amplified_frames[i - 1]  # Motion differences
        motion_sum = np.sum(np.abs(diff))  # Aggregate motion in the frame
        signal.append(motion_sum)
    return np.array(signal)


'''
EVM algorithm
'''


def eulerian_video_magnification(video, ground_truth, fps, window_size,
                                 respiratory_rate_history, frame_processing_times, cpu_loads, mpc, csd, process, x, y, h, w):
    ground_truth = ground_truth * 80000000  # For visualisation purposes, as the motion magnitude is much higher
    respiratory_rate = None
    sliding_window_data = []
    frame_count = 0

    while True:
        start_time = time.time()
        ret, frame = video.read()
        if not ret:
            print("End of video stream.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cropped_frame = gray_frame[y:y + h, x:x + w]
        sliding_window_data.append(cropped_frame)

        frame_count += 1

        if len(sliding_window_data) > window_size:
            sliding_window_data.pop(0)

        if frame_count >= window_size and (frame_count % int(fps) == 0):
            amplified_frames = amplify_motion(sliding_window_data, fps)
            motion_signal = extract_motion_signal(amplified_frames)
            filtered_signal = bandpass_filter(motion_signal, fps, 0.3, 0.8, 8)

            respiratory_rate = fourier(filtered_signal, fps)
            respiratory_rate_history.append(respiratory_rate)

            if frame_count % window_size == 0:
                print("Frame count: ", frame_count)
                # Extract the corresponding ground truth data
                ground_truth_window = ground_truth[(frame_count - window_size):frame_count]

                # Ensure the two arrays are of the same size - For first window plot
                min_length = min(len(ground_truth_window), len(filtered_signal))
                ground_truth_window = ground_truth_window[:min_length]
                filtered_signal = filtered_signal[:min_length]

                mpc.append(hilbert_correlation(ground_truth_window, filtered_signal))
                csd.append(cross_spectral_density_correlation(ground_truth_window, filtered_signal))

                # plot_window(filtered_signal, ground_truth_window, frame_count / fps)

        # Display the respiratory rate on the frame
        if respiratory_rate is not None:
            cv2.putText(frame, f"Respiratory Rate: {respiratory_rate:.2f} bpm", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        cv2.imshow("Respiratory Rate Estimation", frame)

        if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
            break

        # Performance tracking
        elapsed_time = time.time() - start_time
        frame_processing_times.append(elapsed_time)
        cpu_cores = psutil.cpu_count()
        cpu_loads.append(process.cpu_percent(interval=None) / cpu_cores)
