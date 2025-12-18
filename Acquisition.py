import yfinance as yf
import pandas as pd

# CONFIG
TICKER = "GC=F"
PERIOD = "10y"
INTERVAL = "1d"
OUTPUT_FILE = "gold_price_monthly_10y_raw.csv"

print("Mengambil data emas HARIAN dari Yahoo Finance...")

gold = yf.Ticker(TICKER)
df_daily = gold.history(period=PERIOD, interval=INTERVAL)

# BASIC FORMAT
df_daily.reset_index(inplace=True)
df_daily["Date"] = pd.to_datetime(df_daily["Date"], utc=True)
df_daily = df_daily[["Date", "Open", "High", "Low", "Close"]]
df_daily = df_daily.dropna()
df_daily = df_daily.sort_values("Date")
df_daily.set_index("Date", inplace=True)

print(f"Data harian: {len(df_daily)} baris")

# RESAMPLE KE BULANAN
monthly = df_daily.resample("MS").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last"
})

# SAVE RAW MONTHLY DATA
monthly.reset_index().to_csv(OUTPUT_FILE, index=False)
