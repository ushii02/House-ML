import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# ── LOAD & PREVIEW ───────────────────────────────────────────

df = pd.read_csv("house_prices.csv")  
print("Shape:", df.shape)
print(df.head())
print(df.info())


# ── DATA CLEANING ────────────────────────────────────────────

# Drop columns with too many missing values (>40%)
threshold = 0.4 * len(df)
df = df.dropna(thresh=threshold, axis=1)

# Fill remaining numeric nulls with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical nulls with mode
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("\nCleaned shape:", df.shape)
print("Missing values remaining:", df.isnull().sum().sum())


# ══════════════════════════════════════════════════════════════
# PART 1 — REGRESSION: Predict House Price
# ══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("PART 1: REGRESSION")
print("="*50)

# Features & target
target = "SalePrice"  # adjust if your CSV uses a different column name
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLinear Regression:")
print(f"  MAE:  ${mean_absolute_error(y_test, y_pred_lr):,.0f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_lr)):,.0f}")
print(f"  R²:   {r2_score(y_test, y_pred_lr):.3f}")

# --- Random Forest Regressor ---
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)

print("\nRandom Forest Regressor:")
print(f"  MAE:  ${mean_absolute_error(y_test, y_pred_rf):,.0f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_rf)):,.0f}")
print(f"  R²:   {r2_score(y_test, y_pred_rf):.3f}")

# Plot: Actual vs Predicted
plt.figure()
plt.scatter(y_test, y_pred_rf, alpha=0.4, color="#E50914", edgecolors="none")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=1.5)
plt.title("Regression: Actual vs Predicted Sale Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.savefig("reg_actual_vs_predicted.png", dpi=150)
plt.show()

# Feature importance
importances = pd.Series(rf_reg.feature_importances_, index=features)
top10 = importances.nlargest(10)

plt.figure()
sns.barplot(x=top10.values, y=top10.index, palette="Reds_r")
plt.title("Top 10 Most Important Features (Regression)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("reg_feature_importance.png", dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════
# PART 2 — CLASSIFICATION: Expensive vs Affordable
# ══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("PART 2: CLASSIFICATION")
print("="*50)

# Create binary label: 1 = expensive (above median), 0 = affordable
median_price = df[target].median()
df["price_class"] = (df[target] > median_price).astype(int)
print(f"\nMedian price: ${median_price:,.0f}")
print(f"Class distribution:\n{df['price_class'].value_counts()}")

X_clf = df[features]
y_clf = df["price_class"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# --- Logistic Regression ---
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_c, y_train_c)
y_pred_log = log_reg.predict(X_test_c)

print("\nLogistic Regression:")
print(f"  Accuracy: {accuracy_score(y_test_c, y_pred_log):.3f}")
print(classification_report(y_test_c, y_pred_log,
      target_names=["Affordable", "Expensive"]))

# --- Random Forest Classifier ---
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rfc = rf_clf.predict(X_test_c)

print("Random Forest Classifier:")
print(f"  Accuracy: {accuracy_score(y_test_c, y_pred_rfc):.3f}")
print(classification_report(y_test_c, y_pred_rfc,
      target_names=["Affordable", "Expensive"]))

# Confusion matrix
cm = confusion_matrix(y_test_c, y_pred_rfc)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=["Affordable", "Expensive"],
            yticklabels=["Affordable", "Expensive"])
plt.title("Classification: Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("clf_confusion_matrix.png", dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════
# PART 3 — ARIMA: Time Series Price Forecasting
# ══════════════════════════════════════════════════════════════

print("\n" + "="*50)
print("PART 3: ARIMA TIME SERIES")
print("="*50)

# Simulate a monthly time series from YearBuilt + SalePrice
# (In a real dataset with sale dates, group by month instead)
if "YrSold" in df.columns and "MoSold" in df.columns:
    ts_df = df.groupby(["YrSold", "MoSold"])[target].mean().reset_index()
    ts_df["date"] = pd.to_datetime(
        ts_df["YrSold"].astype(str) + "-" + ts_df["MoSold"].astype(str)
    )
    ts_df = ts_df.sort_values("date").set_index("date")
    ts = ts_df[target]
else:
    # Fallback: use YearBuilt average price per year
    ts = df.groupby("YearBuilt")[target].mean().sort_index()
    ts.index = pd.to_datetime(ts.index, format="%Y")

print(f"\nTime series length: {len(ts)} periods")

# Plot the time series
plt.figure()
ts.plot(color="#E50914", linewidth=2)
plt.title("Average House Price Over Time")
plt.ylabel("Average Sale Price ($)")
plt.tight_layout()
plt.savefig("arima_timeseries.png", dpi=150)
plt.show()

# ACF & PACF plots (to choose ARIMA p, d, q)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts, ax=axes[0], lags=min(20, len(ts)//2 - 1))
plot_pacf(ts, ax=axes[1], lags=min(20, len(ts)//2 - 1))
plt.suptitle("ACF & PACF — Used to Choose ARIMA Parameters")
plt.tight_layout()
plt.savefig("arima_acf_pacf.png", dpi=150)
plt.show()

# Fit ARIMA model (p=1, d=1, q=1 is a solid starting point)
model = ARIMA(ts, order=(1, 1, 1))
result = model.fit()
print("\nARIMA Model Summary:")
print(result.summary())

# Forecast next 12 periods
forecast = result.forecast(steps=12)

plt.figure()
ts.plot(label="Historical", color="#221F1F", linewidth=2)
forecast.plot(label="Forecast (12 periods)", color="#E50914",
              linestyle="--", linewidth=2)
plt.title("ARIMA Forecast — Average House Price")
plt.ylabel("Average Sale Price ($)")
plt.legend()
plt.tight_layout()
plt.savefig("arima_forecast.png", dpi=150)
plt.show()

print("\nForecasted values:")
print(forecast.round(2))


# ── FINAL SUMMARY ────────────────────────────────────────────

print("\n" + "="*50)
print("PROJECT COMPLETE — FILES SAVED:")
print("="*50)
print("  reg_actual_vs_predicted.png")
print("  reg_feature_importance.png")
print("  clf_confusion_matrix.png")
print("  arima_timeseries.png")
print("  arima_acf_pacf.png")
print("  arima_forecast.png")