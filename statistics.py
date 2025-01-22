import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

file_path = 'algorithm_results.csv'
column_names = [
    "Algorithm", "Video Path", "FPS", "Avg Frame Processing Time", "Avg CPU Load", "MPC", "CSD", "Pearson Coefficient", "RMSE"
]
df = pd.read_csv(file_path, header=None, names=column_names)

df['Avg Frame Processing Time'] = pd.to_numeric(df['Avg Frame Processing Time'], errors='coerce')
df = df.dropna(subset=['Avg Frame Processing Time'])
df['Avg CPU Load'] = pd.to_numeric(df['Avg CPU Load'], errors='coerce')
df = df.dropna(subset=['Avg CPU Load'])

df['Pearson Coefficient'] = pd.to_numeric(df['Pearson Coefficient'], errors='coerce')
df = df.dropna(subset=['Pearson Coefficient'])
df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')
df = df.dropna(subset=['RMSE'])
df['MPC'] = pd.to_numeric(df['MPC'], errors='coerce')
df = df.dropna(subset=['MPC'])
df['CSD'] = pd.to_numeric(df['CSD'], errors='coerce')
df = df.dropna(subset=['CSD'])

averageFPT = df.groupby('Algorithm')['Avg Frame Processing Time'].mean()
averageCPU = df.groupby('Algorithm')['Avg CPU Load'].mean()

averagePearson = df.groupby('Algorithm')['Pearson Coefficient'].mean()
averageRMSE = df.groupby('Algorithm')['RMSE'].mean()
averageMPC = df.groupby('Algorithm')['MPC'].mean()
averageCSD = df.groupby('Algorithm')['CSD'].mean()

print(averageFPT)
print(averageCPU)

print(averagePearson)
print(averageRMSE)
print(averageMPC)
print(averageCSD)

df['FPS'] = pd.to_numeric(df['FPS'], errors='coerce').astype(float).astype(int)

fps_values = [10, 15, 30]
max_delays = [1 / fps for fps in fps_values]
bar_colours = ["purple", "orange", "green"]

fig, axes = plt.subplots(1, 3, figsize=(15, 8), sharey=True)

plt.rcParams.update({'font.size': 18})

for i, (fps, max_delay) in enumerate(zip(fps_values, max_delays)):
    # Filter data for the current FPS
    df_fps = df[df['FPS'] == fps]

    avg_fpt = df_fps.groupby('Algorithm')['Avg Frame Processing Time'].mean()
    algorithms = avg_fpt.index
    avg_times = avg_fpt.values
    bar_positions = np.arange(len(algorithms))

    axes[i].bar(bar_positions, avg_times, color=bar_colours, label='Avg Frame Processing Time')

    axes[i].axhline(y=max_delay, color='black', linewidth=3, linestyle='--',label=f'Max Delay ({1/fps:.2f}s)')

    axes[i].set_title(f'{fps} FPS')
    axes[i].set_xticks(bar_positions)
    axes[i].set_xticklabels(algorithms, rotation=45, ha='right', fontsize=20)
    axes[i].tick_params(axis='y', labelsize=20)
    axes[i].set_ylabel('Time (s)', fontsize=20) if i == 0 else None

plt.tight_layout()
plt.show()

