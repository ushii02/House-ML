# 🏠 House Price Forecasting — Regression, Classification & ARIMA

Three different modelling approaches on the same real estate dataset.

---

## 📌 What This Project Covers

1. **Regression** — Predict the exact sale price of a house
2. **Classification** — Predict if a house is expensive or affordable
3. **ARIMA** — Forecast how average prices trend over time

---

## 🤖 Models Used

### Part 1 — Regression
- Linear Regression
- Random Forest Regressor
- Evaluated with MAE, RMSE, and R²

### Part 2 — Classification
- Logistic Regression
- Random Forest Classifier
- Evaluated with accuracy, precision, recall, confusion matrix

### Part 3 — Time Series (ARIMA)
- ACF & PACF plots to determine model parameters
- ARIMA(1,1,1) fitted on historical average prices
- 12-period forecast with visualization

---

## 📊 Charts Generated

| Chart | Description |
|-------|-------------|
| reg_actual_vs_predicted.png | Scatter plot of actual vs predicted prices |
| reg_feature_importance.png | Top 10 features driving price predictions |
| clf_confusion_matrix.png | Confusion matrix for classification model |
| arima_timeseries.png | Historical average price trend |
| arima_acf_pacf.png | ACF & PACF plots for ARIMA parameter selection |
| arima_forecast.png | 12-period price forecast |

---

## 💡 Key Findings

- Random Forest significantly outperforms Linear Regression on this dataset
- Overall quality, living area, and neighbourhood are the strongest price predictors
- The classification model achieves high accuracy in separating expensive vs affordable homes
- ARIMA captures the upward price trend and projects it forward

---

## 🛠️ Tools & Libraries

- Python 3.14
- pandas, numpy
- scikit-learn
- statsmodels
- matplotlib, seaborn

---

## 📁 Dataset Source

**House Prices: Advanced Regression Techniques — Kaggle**

The dataset contains ~1,500 residential homes in Ames, Iowa with 80 features describing each property.

---

## ▶️ How to Run

1. Clone this repo
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```

3. Place `train.csv` in the same folder and rename it to `house_prices.csv`
4. Run the script:

```bash
python house_price_ml.py
```
