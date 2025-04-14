import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

# =============================================================================
# 1. Data Loading, Filtering & Basic Diagnostics
# =============================================================================
df = pd.read_csv("/Users/adviti/Desktop/Video_Timeseries_synthetic.csv").dropna().reset_index(drop=True)
print("Original dataset shape:", df.shape)
print(df.info())

print("\nFinal views percentiles:")
print(df['views_final'].quantile([0.01, 0.05, 0.50, 0.95, 0.99]))

# Filter out videos with extremely low final views (set threshold as needed)
min_views = 1000
df = df[df['views_final'] >= min_views].copy()
print("Dataset shape after filtering (views_final >= {}):".format(min_views), df.shape)

# Plot final views distribution (log scale)
plt.figure(figsize=(8,5))
sns.histplot(df['views_final'], bins=50)
plt.xscale("log")
plt.title("Distribution of Final Views (log scale)")
plt.xlabel("Final Views")
plt.ylabel("Frequency")
plt.show()

# =============================================================================
# 2. Create α‑Lifespan Grouping (Two-Stage Approach)
# =============================================================================
alpha = 0.5
df['lifespan_ratio'] = np.where(df['views_final'] > 0, df['views3s_5k'] / df['views_final'], 0)
df['lifespan_group'] = np.where(df['lifespan_ratio'] >= alpha, 'short_lifetime', 'long_lifetime')
print("\nLifespan group counts:")
print(df['lifespan_group'].value_counts())

# =============================================================================
# 3. Normalization and Feature Engineering
# =============================================================================
epsilon = 1e-3
# 3a. Normalized early engagement metrics:
df['norm_views3s_1k'] = df['views3s_1k'] * 1000 / (df['impressions_1k'] + epsilon)
df['norm_like_1k']     = df['like_1k'] * 1000 / (df['impressions_1k'] + epsilon)
df['norm_comments_1k'] = df['comments_1k'] * 1000 / (df['impressions_1k'] + epsilon)
df['norm_views3s_5k']  = df['views3s_5k'] * 1000 / (df['impressions_5k'] + epsilon)
df['norm_like_5k']     = df['like_5k'] * 1000 / (df['impressions_5k'] + epsilon)
df['norm_comments_5k'] = df['comments_5k'] * 1000 / (df['impressions_5k'] + epsilon)

df['norm_views3s_2k'] = df['views3s_2k'] * 1000 / (df['impressions_1k'] + epsilon)
df['norm_views3s_3k'] = df['views3s_3k'] * 1000 / (df['impressions_1k'] + epsilon)
df['norm_views3s_4k'] = df['views3s_4k'] * 1000 / (df['impressions_1k'] + epsilon)

print("\nSample of normalized metrics:")
print(df[['norm_views3s_1k', 'norm_like_1k', 'norm_comments_1k',
          'norm_views3s_5k', 'norm_like_5k', 'norm_comments_5k']].head())
print("\nnorm_views3s_1k stats:")
print(df['norm_views3s_1k'].describe())

# 3b. Engineered rate features.
df['openrate3s_1k']   = np.where(df['impressions_1k'] != 0, df['views3s_1k'] / df['impressions_1k'], 0)
df['openrate3s_5k']   = np.where(df['impressions_5k'] != 0, df['views3s_5k'] / df['impressions_5k'], 0)
df['engrate_1k']      = np.where(df['views3s_1k'] != 0, df['like_1k'] / df['views3s_1k'], 0)
df['engrate_5k']      = np.where(df['views3s_5k'] != 0, df['like_5k'] / df['views3s_5k'], 0)
df['commentratio_1k'] = np.where(df['like_1k'] != 0, df['comments_1k'] / df['like_1k'], 0)
df['commentratio_5k'] = np.where(df['like_5k'] != 0, df['comments_5k'] / df['like_5k'], 0)

# Clip comment ratios to reduce the influence of extreme outliers
df['commentratio_1k'] = df['commentratio_1k'].clip(upper=2)
df['commentratio_5k'] = df['commentratio_5k'].clip(upper=2)

eng_features = ['openrate3s_1k', 'openrate3s_5k', 'engrate_1k', 'engrate_5k',
                'commentratio_1k', 'commentratio_5k']
print("\nEngineered rate features (first 5 rows):")
print(df[eng_features].head())

# Visualize engineered feature distributions.
fig, axes = plt.subplots(len(eng_features), 2, figsize=(12, 4 * len(eng_features)))
for i, feat in enumerate(eng_features):
    sns.histplot(df[feat], bins=50, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f"{feat} (raw)")
    sns.histplot(np.log1p(df[feat]), bins=50, kde=True, ax=axes[i, 1])
    axes[i, 1].set_title(f"{feat} (log1p)")
plt.tight_layout()
plt.show()

# =============================================================================
# 4. Scaling Combined Features and Log-Transforming the Target
# =============================================================================
combined_features = ['norm_views3s_1k', 'norm_like_1k', 'norm_comments_1k',
                     'norm_views3s_2k', 'norm_views3s_3k', 'norm_views3s_4k',
                     'norm_views3s_5k', 'norm_like_5k', 'norm_comments_5k'] + eng_features

scaler = RobustScaler()
X_combined = scaler.fit_transform(df[combined_features].values)
print("\nScaled combined feature matrix shape:", X_combined.shape)

# Log-transform the target.
# df['log_views_final'] = np.log1p(df['views_final'].clip(upper=1e6))
print("Log views (max, mean, std):", df['views_final'].max(), df['views_final'].mean(), df['views_final'].std())

# =============================================================================
# 5. Global Model Evaluation via GridSearchCV
# =============================================================================
from sklearn.model_selection import GridSearchCV

print("\nGlobal Model Evaluation via GridSearchCV:")

models = [
    ('LinearRegression', LinearRegression(), {}),
    ('RandomForest', RandomForestRegressor(random_state=42), {
        'model__n_estimators': [50]
    }),
    ('XGBoost', XGBRegressor(objective='reg:squarederror', random_state=42), {
        'model__n_estimators': [50]
    }),
]

for name, estimator, param_grid in models:
    pipeline = Pipeline([
        ('model', estimator)
    ])
    grid = GridSearchCV(pipeline, param_grid,
                        scoring='neg_mean_absolute_error',
                        refit=True, cv=3, verbose=1)
    grid.fit(X_combined, df['views_final'])

    preds = grid.best_estimator_.predict(X_combined)
    mae_score = mean_absolute_error(df['views_final'], preds)
    mape_score = mean_absolute_percentage_error(df['views_final'], preds) * 100

    print(f"{name}:")
    print(f"   MAE: {mae_score:.2f}")
    print(f"   MAPE: {mape_score:.2f}%")

# =============================================================================
# 6. Clustering and Hybrid Specialized Modeling
# =============================================================================
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Predict lifespan group based on early features
lifespan_features = ['norm_views3s_1k', 'norm_like_1k', 'norm_comments_1k'] + eng_features
clf = GradientBoostingClassifier(random_state=42)
clf.fit(df[lifespan_features], df['lifespan_group'])

df['lifespan_pred'] = clf.predict(df[lifespan_features])
print("\nLifespan class prediction accuracy:")
print((df['lifespan_group'] == df['lifespan_pred']).mean())

# 6b. Lifespan-Specific Model Evaluation via Cross-Validation.
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingRegressor

print("\nLifespan-Specific Model Performance with Tuning and Stacking (5-fold CV):")

param_grids = {
    'short_lifetime': {
        'model': [RandomForestRegressor(random_state=42)],
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 5]
    },
    'long_lifetime': {
        'model': [GradientBoostingRegressor(random_state=42)],
        'model__n_estimators': [50, 100],
        'model__learning_rate': [0.05, 0.1]
    }
}

for group in ['short_lifetime', 'long_lifetime']:
    idx = df['lifespan_pred'] == group
    X_group = X_combined[idx]
    y_group = df['views_final'][idx]

    print(f"\nTuning model for {group} group (N = {np.sum(idx)}):")
    base_grid = GridSearchCV(
        estimator=Pipeline([('model', param_grids[group]['model'][0])]),
        param_grid={k: v for k, v in param_grids[group].items() if k != 'model'},
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=0
    )
    base_grid.fit(X_group, y_group)
    best_base_model = base_grid.best_estimator_

    # Add a simple Stacking Regressor with Ridge
    stack = StackingRegressor(
        estimators=[('base', best_base_model.named_steps['model'])],
        final_estimator=LinearRegression(),
        cv=3
    )

    scores = cross_validate(stack, X_group, y_group, cv=5,
                            scoring='neg_mean_absolute_error',
                            return_train_score=False)

    mae_group = -np.mean(scores['test_score'])
    print(f"{group} MAE (Stacked): {mae_group:.2f}")
