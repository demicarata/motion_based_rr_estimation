import cv2
import numpy as np
import psutil
import time

from helper.filters import bandpass_filter, exponential_moving_average, moving_average_filter, wavelet_denoising
from helper.calculate_RR import fourier
from helper.visualisation import plot_window
from helper.window_correlation import hilbert_correlation, cross_spectral_density_correlation


def optical_flow(video, ground_truth, fps, window_size, respiratory_rate_history,
                 motion_signal, frame_processing_times, cpu_loads, mpc, csd, process, x, y, h, w, prev_gray):
    respiratory_rate = None
    sliding_window_data = []
    frame_count = 0

    while True:
        start_time = time.time()
        ret, frame = video.read()
        if not ret:
            break

        # Extract and convert ROI to grayscale
        roi = frame[y:y + h, x:x + w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


        blurred_prev_gray = cv2.GaussianBlur(prev_gray, (7, 7), 0)
        blurred_roi_gray = cv2.GaussianBlur(roi_gray, (7, 7), 0)

        frame_count += 1

        # Calculate optical flow for the ROI
        flow = cv2.calcOpticalFlowFarneback(blurred_prev_gray, blurred_roi_gray, None,
                                            pyr_scale=0.25, levels=5, winsize=3,
                                            iterations=5, poly_n=5, poly_sigma=1.1, flags=0)

        # Calculate motion magnitude
        fx = flow[..., 0]  # Horizontal motion
        fy = flow[..., 1]  # Vertical motion
        motion_magnitude = np.sqrt(fx ** 2 + fy ** 2)

        # Aggregate motion in the ROI
        motion_value = np.median(motion_magnitude)
        motion_signal.append(motion_value)

        sliding_window_data.append(motion_value)

        # Maintain a rolling buffer of the last 10 seconds
        if len(sliding_window_data) > window_size:
            sliding_window_data = sliding_window_data[-window_size:]

        # Update respiratory rate every 1 second
        if frame_count >= window_size and (frame_count % int(fps) == 0):
            filtered_signal = bandpass_filter(sliding_window_data, fps, 0.3, 0.8, 8)
            filtered_signal = wavelet_denoising(filtered_signal)
            filtered_signal = moving_average_filter(filtered_signal)

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

                plot_window(filtered_signal, ground_truth_window, frame_count/fps)

        # Display the respiratory rate on the frame
        if respiratory_rate is not None:
            cv2.putText(frame, f"Respiratory Rate: {respiratory_rate:.2f} bpm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        cv2.imshow("Respiratory Rate Estimation", frame)

        if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
            break

        prev_gray = roi_gray

        # Performance tracking
        elapsed_time = time.time() - start_time
        frame_processing_times.append(elapsed_time)
        cpu_cores = psutil.cpu_count()
        cpu_loads.append(process.cpu_percent(interval=None) / cpu_cores)
