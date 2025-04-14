import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm
import hashlib

# ------------------
# Logistic Curve Model
# ------------------
def logistic(t, L, k, x0):
    return L / (1 + np.exp(-k * (t - x0)))

# ------------------
# Assign Alpha Lifespan Using LARM Patterns
# ------------------
def assign_alpha_lifespan():
    # Empirical alpha=0.5 lifespan clusters (H1-H6 from LARM Table 3, approx)
    clusters = [2.5, 6, 16, 32, 48, 72]  # hours
    stds = [1, 2, 5, 7, 10, 12]
    cluster = np.random.choice(len(clusters))
    return max(1, np.random.normal(clusters[cluster], stds[cluster]))

# ------------------
# Generate hourly views
# ------------------
def generate_views(final_views, alpha_hour):
    try:
        popt, _ = curve_fit(logistic, [alpha_hour, 672], [0.5 * final_views, final_views], maxfev=10000)
        hours = np.array([1, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168, 672])
        return logistic(hours, *popt)
    except:
        return np.full(12, final_views / 12)  # fallback uniform growth

# ------------------
# Main Pipeline
# ------------------

df = pd.read_csv("/Users/adviti/Downloads/Initial Engagement v1 - _WITH_videos_AS_get_the_videos_.csv")
df = df[~df['impressions_5k'].isna()].copy()

# Anonymize
def anonymize(text):
    return hashlib.sha256(str(text).encode()).hexdigest()[:10]

df['video_id_anon'] = df['video_id'].apply(anonymize)
df['channel_id_anon'] = df['channel_id'].apply(anonymize)
df.drop(columns=['video_id', 'channel_id', 'video_published_timestamp', 'creator_type_id'], errors='ignore', inplace=True)

# Time windows
time_hours = [1, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168, 672]

# Generate synthetic decay
rows = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    alpha_hour = assign_alpha_lifespan()
    views = generate_views(row['views_final'], alpha_hour)
    impressions = generate_views(row['impressions_final'], alpha_hour)
    likes = generate_views(row['like_final'], alpha_hour)
    comments = generate_views(row['comment_final'], alpha_hour)

    base = {
        'video_id_anon': row['video_id_anon'],
        'channel_id_anon': row['channel_id_anon'],
        'video_length': row['video_length'],
        'views_final': row['views_final'],
        'impressions_final': row['impressions_final'],
        'like_final': row['like_final'],
        'comment_final': row['comment_final'],
        'alpha_hour': alpha_hour
    }
    for i, h in enumerate(time_hours):
        base[f'views_{h}h'] = views[i]
        base[f'impressions_{h}h'] = impressions[i]
        base[f'likes_{h}h'] = likes[i]
        base[f'comments_{h}h'] = comments[i]

    rows.append(base)

# Output
final_df = pd.DataFrame(rows)
final_df.to_csv("synthetic_time_windows_realistic.csv", index=False)
print("✅ Saved to synthetic_time_windows_realistic.csv")
