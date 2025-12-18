import yfinance as yf
import pandas as pd

# CONFIG
TICKER = "GC=F"
PERIOD = "10y"
INTERVAL = "1d"
OUTPUT_FILE = "gold_price_monthly_10y_clean.csv"
print("Mengambil data emas HARIAN dari Yahoo Finance...")

gold = yf.Ticker(TICKER)
df_daily = gold.history(period=PERIOD, interval=INTERVAL)

# BASIC CLEAN
df_daily.reset_index(inplace=True)
df_daily["Date"] = pd.to_datetime(df_daily["Date"], utc=True)
df_daily = df_daily[["Date", "Open", "High", "Low", "Close"]]
df_daily = df_daily.dropna()
df_daily = df_daily.sort_values("Date")

# index time series
df_daily.set_index("Date", inplace=True)

print(f"Data harian: {len(df_daily)} baris")

# RESAMPLE KE BULANAN
monthly = df_daily.resample("MS").agg({
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last"
})

# CEK MISSSING DATES 
full_month_index = pd.date_range(
    start=monthly.index.min(),
    end=monthly.index.max(),
    freq="MS"
)

monthly = monthly.reindex(full_month_index)
monthly["Close"] = monthly["Close"].interpolate(method="linear")
monthly["Open"]  = monthly["Open"].fillna(monthly["Close"])
monthly["High"]  = monthly["High"].fillna(monthly["Close"])
monthly["Low"]   = monthly["Low"].fillna(monthly["Close"])

monthly.reset_index(inplace=True)
monthly.rename(columns={"index": "Date"}, inplace=True)
monthly_final = monthly[["Date", "Close"]]
monthly_final.rename(columns={"Close": "Price"}, inplace=True)

# SAVE CSV 
monthly_final.to_csv(OUTPUT_FILE, index=False)
