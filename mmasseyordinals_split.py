"""
Extremely large file size of MMasseyOrdinals.csv will cause read/filter
issues down the road. This script plans to open this file using polars, and 
create a different CSV for each ranking system listed within the main file.
"""

import polars 
import os

# Create new subdirectory within data called ankings
os.makedirs('data/rankings')

# Split out each system name into a different file
data = polars.read_csv('data/MMasseyOrdinals.csv')
for sysName in data["SystemName"].unique():
    df = data.filter(
        polars.col("SystemName") == sysName
    )
    df.write_csv(f'data/rankings/MMasseyOrdinals_{sysName}.csv')

# Remove master CSV file
os.remove('data/MMasseyOrdinals.csv')