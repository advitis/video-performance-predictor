import pandas as pd
import hashlib
df = pd.read_csv('/Users/adviti/code/advitis/video-performance-predictor/data/Original_Video_Time_Series.csv')
print("Initial DataFrame shape:", df.shape)

import os
print(os.getcwd())

def hash_value(x):
    x = str(x)
    return hashlib.sha256(x.encode('utf-8')).hexdigest()

df['video_id'] = df['video_id'].astype(str).apply(hash_value)
df['channel_id'] = df['channel_id'].astype(str).apply(hash_value)

columns_to_drop = ['video_published_timestamp', 'channel_name', 'creator_type_id']
df.drop(columns=columns_to_drop, inplace=True)

print("DataFrame shape after dropping columns:", df.shape)

df.to_csv('/Users/adviti/code/advitis/video-performance-predictor/data/Video_Timeseries.csv', index=False)
