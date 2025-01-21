import pandas as pd


file_path = 'algorithm_results.csv'
column_names = [
    'Algorithm', 'Video Path', 'FPS', 'Avg Frame Processing Time', 'Pearson Coefficient', 'RMSE'
]
df = pd.read_csv(file_path, header=None, names=column_names)

df['Avg Frame Processing Time'] = pd.to_numeric(df['Avg Frame Processing Time'], errors='coerce')
df = df.dropna(subset=['Avg Frame Processing Time'])
df['Pearson Coefficient'] = pd.to_numeric(df['Pearson Coefficient'], errors='coerce')
df = df.dropna(subset=['Pearson Coefficient'])
df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')
df = df.dropna(subset=['RMSE'])

averageFPT = df.groupby('Algorithm')['Avg Frame Processing Time'].mean()
averagePearson = df.groupby('Algorithm')['Pearson Coefficient'].mean()
averageRMSE = df.groupby('Algorithm')['RMSE'].mean()

print(averageFPT)
print(averagePearson)
print(averageRMSE)

