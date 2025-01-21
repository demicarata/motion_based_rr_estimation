import cv2
import numpy as np
import time
import psutil
from scipy.stats import pearsonr

from helper.filters import bandpass_filter, exponential_moving_average, median_filter, wavelet_denoising, \
    gaussian_filter, savgol, magnitude_threshold_filter
from helper.calculate_RR import fourier
from helper.visualisation import plot_window


def pixel_intensity_changes(video, ground_truth, fps, window_size, motion_threshold, respiratory_rate_history,
                            motion_signal, frame_processing_times, cpu_loads, process, x, y, h, w):
    ground_truth = 10 * ground_truth
    respiratory_rate = None
    sliding_window_data = []
    prev_frame = None
    last_valid_intensity = None
    frame_count = 0
    motion_differences = []
    k = 3.0
    adaptive_threshold = 100.0


    while True:
        start_time = time.time()
        ret, frame = video.read()

        if not ret:
            print("Error: Can't read frame")
            break

        roi_frame = frame[y:y + h, x:x + w]

        # Display the ROI separately
        cv2.imshow("ROI View", roi_frame)

        frame_count += 1

        # Split signal into RGB values
        (B, G, R) = cv2.split(roi_frame)

        # Compute intensity changes
        avg_intensity_per_line = []
        for line in range(h):
            intensity = np.mean(R[line, :] + G[line, :] + B[line, :])  # Sum RGB and average
            avg_intensity_per_line.append(intensity)

        # Store intensity for this frame
        avg_intensity_per_line = np.array(avg_intensity_per_line)
        if prev_frame is not None:
            # Compute the absolute difference between the current and previous frame
            frame_diff = cv2.absdiff(prev_frame, roi_frame)
            mean_diff = np.mean(frame_diff)
            # print("Frame diff: ", mean_diff)
            motion_differences.append(mean_diff)

            if frame_count % (fps * 10) == 0:
                mean_motion = np.mean(motion_differences)
                std_motion = np.std(motion_differences)
                adaptive_threshold = mean_motion + k * std_motion
                print(mean_motion, std_motion, adaptive_threshold)

            # If the mean difference exceeds the threshold, discard the frame
            if mean_diff > adaptive_threshold:
                print("Significant motion detected, re-adding last valid frame")
                # Append the last valid intensity to maintain sample consistency
                motion_signal.append(last_valid_intensity)
                sliding_window_data.append(last_valid_intensity)

            else:
                motion_signal.append(avg_intensity_per_line)
                sliding_window_data.append(avg_intensity_per_line)

        # Update last valid intensity
        last_valid_intensity = avg_intensity_per_line

        # Keep only the last 10 seconds of data in the sliding window
        if len(sliding_window_data) > window_size:
            sliding_window_data = sliding_window_data[-window_size:]

        # Calculate respiratory rate every second after the first 10 seconds
        if len(sliding_window_data) >= window_size and (frame_count % int(fps) == 0):

            # Convert intensity to a NumPy array
            intensity_window = np.array(sliding_window_data)

            # Select top 5% of lines based on std
            std_devs = np.std(intensity_window, axis=0)
            top_5 = np.argsort(std_devs)[-int(0.05 * len(std_devs)):]
            selected_signal = np.mean(intensity_window[:, top_5], axis=1)

            filtered_signal = bandpass_filter(selected_signal, fps, 0.3, 0.8)

            # filtered_signal = wavelet_denoising(filtered_signal)
            filtered_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
            filtered_signal = exponential_moving_average(filtered_signal)
            # filtered_signal = magnitude_threshold_filter(filtered_signal)

            respiratory_rate = fourier(filtered_signal, fps)
            respiratory_rate_history.append(respiratory_rate)

            if frame_count % (fps * 10) == 0:
                # Extract the corresponding ground truth data
                ground_truth_window = ground_truth[(frame_count - window_size):frame_count]
                # Compute the Mean Squared Error (MSE)
                mse = np.mean((filtered_signal - ground_truth_window) ** 2)

                # Compute the Normalized Root Mean Squared Error (NRMSE)
                range_gt = np.max(ground_truth_window) - np.min(ground_truth_window)
                nrmse = np.sqrt(mse) / range_gt

                print("Normalized Root Mean Squared Error (NRMSE):", nrmse)

                corr_coef, p_value = pearsonr(ground_truth_window, filtered_signal)
                print("Pearson correlation coefficient in window:", corr_coef)

                # print(filtered_signal)

                plot_window(filtered_signal, ground_truth_window, frame_count/fps)

        # Display the respiratory rate on the frame
        if respiratory_rate is not None:
            cv2.putText(frame, f"Respiratory Rate: {respiratory_rate:.2f} bpm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Respiratory Rate Estimation", frame)
        prev_frame = roi_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Performance tracking
        elapsed_time = time.time() - start_time
        frame_processing_times.append(elapsed_time)
        cpu_cores = psutil.cpu_count()
        cpu_loads.append(process.cpu_percent(interval=None) / cpu_cores)
