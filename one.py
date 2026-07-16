import pandas as pd
import numpy as np

# 1. Load the Excel file and target column
file_path = "Train.xls"  # Replace with your actual file name
column_name = "Cost_of_the_Product"  # Replace with your actual column name

# Use 'usecols' to efficiently import only the column you need
df = pd.read_csv(file_path, usecols=[column_name])

# 2. Convert the column to a NumPy array, dropping any empty (NaN) cells
data_array = df[column_name].dropna().to_numpy()

print(data_array)	

x=np.where(data_array<0)
print(x)
