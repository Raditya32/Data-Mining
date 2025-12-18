import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# SETTING
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

# LOAD DATA
df = pd.read_csv("gold_price_monthly_10y_preprocessed.csv")

# DATE FORMAT 
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("===== HEAD DATA =====")
print(df.head())

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe().T)

# TREND HARGA EMAS BULANAN
plt.figure()
plt.plot(df["date"], df["price"], linewidth=2)
plt.title("Trend Harga Emas Bulanan (10 Tahun)")
plt.xlabel("Tahun")
plt.ylabel("Harga Emas (USD)")
plt.grid(True)
plt.show()

# MOVING AVERAGE
plt.figure()
plt.plot(df["date"], df["price"], label="Harga Emas", alpha=0.7)
plt.plot(df["date"], df["ma_3"], label="MA 3 Bulan", linestyle="--")
plt.plot(df["date"], df["ma_12"], label="MA 12 Bulan", linestyle="--")
plt.title("Harga Emas dan Moving Average")
plt.xlabel("Tahun")
plt.ylabel("Harga Emas (USD)")
plt.legend()
plt.grid(True)
plt.show()

# DISTRIBUSI HARGA
plt.figure()
sns.histplot(df["price"], bins=25, kde=True)
plt.title("Distribusi Harga Emas Bulanan")
plt.xlabel("Harga Emas (USD)")
plt.show()

# HEATMAP KORELASI
plt.figure(figsize=(8, 6))

corr = df[["price", "ma_3", "ma_12"]].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.5
)

plt.title("Heatmap Korelasi Fitur Harga Emas Bulanan")
plt.show()
