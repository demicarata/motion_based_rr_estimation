import argparse
import csv
import os
import cv2
import h5py
import numpy as np
import psutil
from scipy.stats import pearsonr

from algorithms.evm import eulerian_video_magnification
from algorithms.of import optical_flow
from algorithms.pic import pixel_intensity_changes
from helper.ground_truth import ground_truth_rr
from helper.visualisation import plot_performance_metrics, plot_ground_truth_rr


def main():
    parser = argparse.ArgumentParser(description="Run video analysis using different algorithms.")
    parser.add_argument("--algorithm", type=int, choices=[1, 2, 3], default=3,
                        help="Select the algorithm: 1 for Pixel Intensity Changes,"
                             " 2 for Optical Flow, 3 for Eulerian Video Magnification.")

    args = parser.parse_args()

    choice = args.algorithm
    print(f"Using algorithm {choice}")

    video_path = "AIR_converted/S05/videos/005_720p.mp4"
    ground_truth_file = "AIR_converted/S05/hdf5/005.hdf5"

    with h5py.File(ground_truth_file, 'r') as f:
        ground_truth = f['respiration'][:]

    if "S06" in ground_truth_file or "S08" in ground_truth_file:
        ground_truth = 5 * ground_truth

    print(f"Length of the ground truth data: {len(ground_truth)}")

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_number = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_number / fps
    window_size = int(fps * 15)
    frame_delay = 1 / fps

    print(f"Frame number: {frame_number}")
    print(f"Duration: {duration}")
    print(f"FPS: {fps}")

    if not video.isOpened():
        print("Error opening video stream or file")
        exit()

    # Select ROI
    ret, first_frame = video.read()
    if not ret:
        print("Error: Can't read the first frame")
        video.release()
        exit()

    roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = map(int, roi)

    motion_signal = []
    respiratory_rate_history = []
    mpc = []
    csd = []

    # Performance tracking
    frame_processing_times = []
    cpu_loads = []
    process = psutil.Process()

    prev_gray = cv2.cvtColor(first_frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)

    if choice == 1:
        algorithm_name = "Pixel Intensity Changes"
        pixel_intensity_changes(video, ground_truth, fps, window_size, respiratory_rate_history,
                                motion_signal, frame_processing_times, cpu_loads, mpc, csd, process, x, y, h, w)
    elif choice == 2:
        algorithm_name = "Optical Flow"
        optical_flow(video, ground_truth, fps, window_size, respiratory_rate_history, motion_signal,
                     frame_processing_times, cpu_loads, mpc, csd, process, x, y, h, w, prev_gray)
    else:
        algorithm_name = "Eulerian Video Magnification"
        eulerian_video_magnification(video, ground_truth, fps, window_size, respiratory_rate_history,
                                     frame_processing_times, cpu_loads, mpc, csd, process, x, y, h, w)

    video.release()
    cv2.destroyAllWindows()

    # Plot performance and accuracy
    plot_performance_metrics(frame_processing_times, frame_delay, cpu_loads)

    ground_truth_bpm = ground_truth_rr(ground_truth_file, int(duration), int(fps))
    plot_ground_truth_rr(ground_truth_bpm, respiratory_rate_history)

    min_length = len(respiratory_rate_history)
    ground_truth_bpm = ground_truth_bpm[:min_length]

    corr_coef, p_value = pearsonr(ground_truth_bpm, respiratory_rate_history)
    print("Pearson correlation coefficient:", corr_coef)
    print("p-value:", p_value)

    differences = np.array(ground_truth_bpm) - np.array(respiratory_rate_history)
    squared_differences = differences ** 2
    rmse = np.sqrt(np.mean(squared_differences))
    print("Root Mean Squared Error (RMSE):", rmse)

    # Save results to CSV
    csv_file = "algorithm_results.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header if file is created for the first time
            writer.writerow(
                ["Algorithm", "Video Path", "FPS", "Avg Frame Processing Time", "Avg CPU Load", "MPC", "CSD", "Pearson Coefficient", "RMSE"])
        writer.writerow([
            algorithm_name,
            video_path,
            fps,
            np.mean(frame_processing_times),
            np.mean(cpu_loads),
            np.mean(mpc),
            np.mean(csd),
            corr_coef,
            rmse
        ])

    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    main()
