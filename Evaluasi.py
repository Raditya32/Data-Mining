import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf

# LOAD MODEL & DATA
with open("sarima_model.pkl", "rb") as f:
    data = pickle.load(f)

result = data["result"]
test   = data["test"]

# PREDIKSI TEST SET
pred_log_test = result.predict(
    start=test.index[0],
    end=test.index[-1],
    dynamic=False
)

pred_price_test = np.exp(pred_log_test)
actual_price_test = test["price"]

# METRIK EVALUASI
rmse = np.sqrt(mean_squared_error(actual_price_test, pred_price_test))
mape = mean_absolute_percentage_error(actual_price_test, pred_price_test)

print("\n===== EVALUASI MODEL SARIMA =====")
print(f"RMSE : {rmse:.2f} USD")
print(f"MAPE : {mape*100:.2f} %")

# VISUALISASI: AKTUAL vs PREDIKSI
plt.figure(figsize=(14, 6))
plt.plot(actual_price_test.index, actual_price_test,
         label="Harga Aktual", linewidth=2)
plt.plot(pred_price_test.index, pred_price_test,
         label="Harga Prediksi SARIMA", linestyle="--", linewidth=2)

plt.title("Perbandingan Harga Aktual vs Prediksi (Test Set)")
plt.xlabel("Waktu")
plt.ylabel("Harga Emas (USD)")
plt.legend()
plt.grid(True)
plt.show()

# ANALISIS RESIDUAL
residuals = actual_price_test - pred_price_test

plt.figure(figsize=(14, 5))
plt.plot(residuals.index, residuals, label="Residual", color="red")
plt.axhline(0, linestyle="--")
plt.title("Residual Time Series")
plt.xlabel("Waktu")
plt.ylabel("Error (USD)")
plt.legend()
plt.grid(True)
plt.show()

# DISTRIBUSI RESIDUAL
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=20)
plt.title("Distribusi Residual")
plt.xlabel("Residual")
plt.ylabel("Frekuensi")
plt.grid(True)
plt.show()

# POLA KESALAHAN (ACF RESIDUAL)
plt.figure(figsize=(10, 4))
plot_acf(residuals.dropna(), lags=24)
plt.title("ACF Residual (Deteksi Pola Error)")
plt.show()

# STABILITAS PREDIKSI
rolling_rmse = []

window = 6  # 6 bulan
for i in range(window, len(actual_price_test)):
    rmse_i = np.sqrt(
        mean_squared_error(
            actual_price_test.iloc[i-window:i],
            pred_price_test.iloc[i-window:i]
        )
    )
    rolling_rmse.append(rmse_i)

rolling_rmse = pd.Series(
    rolling_rmse,
    index=actual_price_test.index[window:]
)

plt.figure(figsize=(14, 5))
plt.plot(rolling_rmse.index, rolling_rmse, linewidth=2)
plt.title("Rolling RMSE (Stabilitas Prediksi Model)")
plt.xlabel("Waktu")
plt.ylabel("RMSE")
plt.grid(True)
plt.show()
