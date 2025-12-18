import pandas as pd
import numpy as np

# LOAD DATA
df = pd.read_csv("gold_price_monthly_10y_clean.csv")
df.columns = df.columns.str.lower().str.strip()

# DATE CLEAN
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# time series
df.set_index('date', inplace=True)
df = df.asfreq('MS')

# TARGET
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# HANDLE MISSING
df = df.replace(['', ' ', 'NA', 'N/A', 'null'], np.nan)
df['price'] = df['price'].interpolate(method='linear')

# FEATURE ENGINEERING
df['ma_3']  = df['price'].rolling(window=3, min_periods=1).mean()
df['ma_12'] = df['price'].rolling(window=12, min_periods=1).mean()
df = df.dropna().reset_index()

# SAVE 
OUTPUT_FILE = "gold_price_monthly_10y_preprocessed.csv"
df.to_csv(OUTPUT_FILE, index=False)

print("\n===== PREVIEW =====")
print(df.head())

print("\n===== INFO =====")
print(df.info())
