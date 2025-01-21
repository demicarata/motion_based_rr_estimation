import cv2
import numpy as np
import psutil
import time
from helper.filters import bandpass_filter, exponential_moving_average, moving_average_filter, wavelet_denoising
from helper.calculate_RR import fourier
from helper.visualisation import plot_window


def optical_flow(video, ground_truth, fps, window_size, respiratory_rate_history,
                 motion_signal, frame_processing_times, cpu_loads, process, x, y, h, w, prev_gray):
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
        if len(sliding_window_data) == window_size and (frame_count % int(fps) == 0):
            smoothed_signal = bandpass_filter(sliding_window_data, fps, 0.3, 0.8)
            smoothed_signal = wavelet_denoising(smoothed_signal)
            smoothed_signal  = moving_average_filter(smoothed_signal)

            respiratory_rate = fourier(smoothed_signal, fps)
            respiratory_rate_history.append(respiratory_rate)

            if frame_count % (fps * 10) == 0:
                print("Frame count: ", frame_count)
                # Extract the corresponding ground truth data
                ground_truth_window = ground_truth[(frame_count - window_size):frame_count]

                plot_window(smoothed_signal, ground_truth_window, frame_count/fps)

        # Display the respiratory rate on the frame
        if respiratory_rate is not None:
            cv2.putText(frame, f"Respiratory Rate: {respiratory_rate:.2f} bpm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        cv2.imshow("Respiratory Rate Estimation", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
            break

        prev_gray = roi_gray

        # Performance tracking
        elapsed_time = time.time() - start_time
        frame_processing_times.append(elapsed_time)
        cpu_cores = psutil.cpu_count()
        cpu_loads.append(process.cpu_percent(interval=None) / cpu_cores)
