# 🎯 Predicting Video Performance from Early Engagement

This project is a simplified demonstration of how early engagement metrics (e.g. views, likes, comments, impressions) can be used to predict a video’s long-term performance on social platforms.

It showcases a full modeling pipeline — from data cleaning to feature engineering to evaluation — while intentionally working with a very limited set of early metrics to reflect real-world constraints and business urgency.

All analysis is in the notebook: **[`video_performance_predictor.ipynb`](notebooks/video_performance_predictor.ipynb)**

---

## 📊 Key Result

- **Model:** Random Forest Regressor
- **Performance:** MAPE ≈ 36%, MAE ≈ 96K
- **Dataset:** Synthetic, 5,100 videos (simulating real-world distributions)

---

## 💼 Business Use Cases

- Spot high-potential videos early
- Allocate promotion budget intelligently
- Guide creator and content strategy

---

## 🛠️ Tech Stack

- Python, pandas, scikit-learn, xgboost
- Modeling: Linear Regression, Random Forest, XGBoost
- Evaluation: Cross-validation, MAE, MAPE

---

## 🚀 Quick Start

Clone the repo and run the notebook:

```bash
pip install -r requirements.txt
jupyter notebook
