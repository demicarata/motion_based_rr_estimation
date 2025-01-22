from matplotlib import pyplot as plt


def plot_window(signal, ground_truth, time_stamp):
    # Currently the ground truth keeps getting smaller? - What
    plt.figure(figsize=(12, 6))

    # Plot the filtered motion signal
    plt.plot(signal, label="Filtered Motion Signal", color="purple")

    # Plot the ground truth data
    plt.plot(ground_truth, label="Ground Truth Respiration", color="orange", linestyle="--")

    plt.title(f"Filtered Motion Signal and Ground Truth at {time_stamp:.1f} seconds", fontsize=20)
    plt.xlabel("Time (frame number)", fontsize=20)
    plt.ylabel("Normalized Intensity / Respiration", fontsize=20)
    plt.rcParams['font.size'] = 20
    plt.subplots_adjust(bottom=0.2)
    plt.legend()
    plt.show()


def plot_performance_metrics(frame_processing_times, frame_delay, cpu_loads):
    plt.figure(figsize=(12, 8))

    # Plot the frame processing times in comparison with the threshold
    plt.subplot(2, 1, 2)
    plt.plot(frame_processing_times, label="Frame Processing Time (s)", color="blue")
    plt.axhline(frame_delay, color="red", linestyle="--", label="Frame Processing Threshold")
    plt.title("Frame Processing Times", fontsize=18)
    plt.xlabel("Frame (n)", fontsize=18)
    plt.ylabel("Processing Time (s)", fontsize=18)
    plt.legend()

    # Plot CPU load and frame processing time
    plt.subplot(2, 1, 1)
    plt.plot(cpu_loads, label="CPU Load (%)", color="green")
    plt.title("CPU Load Over Time", fontsize=18)
    plt.xlabel("Frame (n)", fontsize=18)
    plt.ylabel("CPU Load (%)", fontsize=18)

    plt.rcParams['font.size'] = 18
    plt.tight_layout()
    plt.show()


def plot_ground_truth_rr(ground_truth, respiratory_rate_history):
    # Fix the Time label not being shown completely
    plt.plot(ground_truth, label="Ground Truth", color="blue")
    plt.plot(respiratory_rate_history, label="Extracted Respiratory Rate", color="pink")
    plt.title("Extracted RR vs Ground Truth")
    plt.xlabel("Time(s)")
    plt.ylabel("RR(BPM)")
    plt.legend()
    plt.rcParams['font.size'] = 18
    plt.subplots_adjust(bottom=0.2)
    plt.show()
