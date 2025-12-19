import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# LOAD DATA
df = pd.read_csv("gold_price_monthly_10y_preprocessed.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# TIME SERIES BULANAN
df.set_index("date", inplace=True)
df = df.asfreq("MS")

# TARGET
df["price"] = df["price"].interpolate(method="linear")

# LOG TRANSFORM
df["log_price"] = np.log(df["price"])

# ADF TEST (LEVEL)
print("\n===== ADF TEST =====")
adf_level = adfuller(df["log_price"].dropna())

print(f"ADF Statistic : {adf_level[0]:.4f}")
print(f"p-value       : {adf_level[1]:.4f}")
print("Critical Values:")
for key, value in adf_level[4].items():
    print(f"   {key} : {value:.4f}")

if adf_level[1] <= 0.05:
    print("Kesimpulan : Data STASIONER (tolak H0)")
else:
    print("Kesimpulan : Data TIDAK stasioner (gagal tolak H0)")

# DIFFERENCING
diff_log_price = df["log_price"].diff().dropna()

# ADF TEST
print("\n===== ADF TEST (LOG PRICE - FIRST DIFFERENCING) =====")
adf_diff = adfuller(diff_log_price)

print(f"ADF Statistic : {adf_diff[0]:.4f}")
print(f"p-value       : {adf_diff[1]:.4f}")
print("Critical Values:")
for key, value in adf_diff[4].items():
    print(f"   {key} : {value:.4f}")

if adf_diff[1] <= 0.05:
    print("Kesimpulan : Data STASIONER setelah differencing")
else:
    print("Kesimpulan : Data MASIH tidak stasioner")

# ACF & PACF
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plot_acf(diff_log_price, lags=24, ax=ax)
ax.set_title("ACF Log-Price (After First Differencing)")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plot_pacf(diff_log_price, lags=24, ax=ax, method="ywm")
ax.set_title("PACF Log-Price (After First Differencing)")
plt.show()

# TRAIN / TEST SPLIT
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

# SARIMA PARAMETER SEARCH (AIC)
import warnings
warnings.filterwarnings("ignore")

best_aic = np.inf
best_order = None
best_seasonal_order = None

p = q = range(0, 3)
d = [1]
P = Q = range(0, 2)
D = [1]
s = 12

for order in [(i, d[0], j) for i in p for j in q]:
    for seasonal_order in [(i, D[0], j, s) for i in P for j in Q]:
        try:
            model = SARIMAX(
                train["log_price"],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            result = model.fit(disp=False)

            if result.aic < best_aic:
                best_aic = result.aic
                best_order = order
                best_seasonal_order = seasonal_order
        except:
            continue

print("\n===== BEST SARIMA PARAMETER =====")
print(f"Order            : {best_order}")
print(f"Seasonal Order   : {best_seasonal_order}")
print(f"AIC              : {best_aic:.2f}")

# SARIMA MODEL
model = SARIMAX(
    train["log_price"],
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

result = model.fit(disp=False)
print(result.summary())

# FINAL MODEL
final_model = SARIMAX(
    df["log_price"],
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

final_result = final_model.fit(disp=False)

# FORECAST 2026
n_forecast = 12

forecast_log = final_result.forecast(steps=n_forecast)
forecast_price = np.exp(forecast_log)

forecast_dates = pd.date_range(
    start="2026-01-01",
    periods=n_forecast,
    freq="MS"
)

forecast_df = pd.DataFrame({
    "date": forecast_dates,
    "predicted_gold_price": forecast_price.values
})

forecast_df.to_csv(
    "gold_price_forecast_2026_SARIMA_MONTHLY.csv",
    index=False
)

# VISUALISASI
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["price"], label="Historical Price", linewidth=2)
plt.plot(
    forecast_df["date"],
    forecast_df["predicted_gold_price"],
    linestyle="--",
    marker="o",
    linewidth=2,
    label="Forecast 2026"
)

plt.title("Prediksi Harga Emas Bulanan 2026 (SARIMA)")
plt.xlabel("Tahun")
plt.ylabel("Harga Emas (USD)")
plt.legend()
plt.grid(True)
plt.show()
