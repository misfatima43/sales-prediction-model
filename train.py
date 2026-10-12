from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
rng = np.random.default_rng(7)
n = 900
day = np.arange(n)
df = pd.DataFrame({
    "day_index": day,
    "tv_spend": rng.uniform(0, 300, n),
    "digital_spend": rng.uniform(0, 150, n),
    "discount_pct": rng.uniform(0, 0.35, n),
    "unit_price": rng.uniform(8, 35, n),
    "is_weekend": (day % 7 >= 5).astype(int),
})
df["month_sin"] = np.sin(2 * np.pi * (day % 365) / 365)
df["month_cos"] = np.cos(2 * np.pi * (day % 365) / 365)
df["sales"] = (220 + 2.4 * df.tv_spend + 3.1 * df.digital_spend
               + 420 * df.discount_pct - 8.5 * df.unit_price
               + 65 * df.is_weekend + 35 * df.month_sin
               + rng.normal(0, 45, n)).clip(lower=0)
df.to_csv(ROOT / "synthetic_sales.csv", index=False)

# A chronological holdout is used to reduce leakage for this time-ordered demo.
features = ["tv_spend", "digital_spend", "discount_pct", "unit_price",
            "is_weekend", "month_sin", "month_cos"]
split = int(len(df) * 0.8)
X_train, X_test = df[features].iloc[:split], df[features].iloc[split:]
y_train, y_test = df["sales"].iloc[:split], df["sales"].iloc[split:]
models = {"LinearRegression": LinearRegression(),
          "RandomForest": RandomForestRegressor(n_estimators=180, random_state=7, min_samples_leaf=3)}
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({"model": name, "MAE": mean_absolute_error(y_test, pred),
                 "RMSE": mean_squared_error(y_test, pred) ** 0.5, "R2": r2_score(y_test, pred)})
results = pd.DataFrame(rows).sort_values("RMSE")
results.to_csv(ROOT / "evaluation_results.csv", index=False)
print(results.to_string(index=False))
print("Files created: synthetic_sales.csv, evaluation_results.csv")
