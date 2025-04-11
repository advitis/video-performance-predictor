# Dataset Simulation Logic

This project uses a statistically grounded synthetic dataset inspired by real-world video performance data on social platforms.

## Objective

To model final video performance (views_final, like_final, etc.) using early engagement signals collected at 1k and 5k impression milestones.

⸻

## Dataset Features
	•	video_id, channel_name: Randomly generated identifiers
	•	video_length: Randomized (15s to 1800s)
	•	1k Milestone Metrics (always present):
	•	views3s_1k, like_1k, comments_1k, impressions_1k = 1000
	•	5k Milestone Metrics (65% of videos):
	•	views3s_5k, like_5k, comments_5k, impressions_5k = 5000
	•	For videos that didn’t reach 5k, capped synthetic estimates are generated
	•	Final Outcome Metrics:
	•	views_final, impressions_final, like_final, comment_final

⸻

## Notes
	•	All metrics follow realistic engagement ratios (CTR, like rates, comment rates)
	•	All rows are synthetic: no real data reused
	•	No missing values — blank 5k milestones are plausibly filled using early engagement
