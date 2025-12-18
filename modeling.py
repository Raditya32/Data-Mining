import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# LOAD DATA
df = pd.read_csv("gold_price_monthly_10y_preprocessed.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# time series bulanan
df.set_index("date", inplace=True)
df = df.asfreq("MS")

# TARGET 
df["price"] = df["price"].interpolate(method="linear")

# LOG TRANSFORM
df["log_price"] = np.log(df["price"])

# ACF & PACF
diff_log_price = df["log_price"].diff().dropna()

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plot_acf(diff_log_price, lags=24, ax=ax)
ax.set_title("ACF Log-Price (First Differencing)")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plot_pacf(diff_log_price, lags=24, ax=ax, method="ywm")
ax.set_title("PACF Log-Price (First Differencing)")
plt.show()

# TRAIN / TEST SPLIT 
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

# SARIMA
model = SARIMAX(
    train["log_price"],
    order=(1, 1, 1),              
    seasonal_order=(1, 1, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

result = model.fit(disp=False)
print(result.summary())

# EVALUASI
pred_log_test = result.predict(
    start=test.index[0],
    end=test.index[-1],
    dynamic=False
)

pred_price_test = np.exp(pred_log_test)
actual_price_test = test["price"]

rmse = np.sqrt(mean_squared_error(actual_price_test, pred_price_test))
mape = mean_absolute_percentage_error(actual_price_test, pred_price_test)

print("\n===== EVALUASI MODEL =====")
print(f"RMSE : {rmse:.2f} USD")
print(f"MAPE : {mape*100:.2f}%")

# FINAL MODEL 
final_model = SARIMAX(
    df["log_price"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 12),
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
