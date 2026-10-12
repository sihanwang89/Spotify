# 🎵 Global Spotify Hits — Song Longevity & Audio Dynamics (2018–2026)

## Overview

Why do some hit songs disappear after two weeks while others remain on streaming charts for years? This dataset combines song metadata, detailed audio features, and chart longevity metrics to help data scientists model music trends and viral dynamics.

## Dataset Structure

- `track_metadata.csv` (1,200 rows): Basic release details and genre tags.
- `audio_features.csv` (1,200 rows): Quantitative musical characteristics (Tempo, Energy, Valence, Danceability).
- `chart_longevity_panel.csv` (1,200 rows): Chart positions, stream counts, and target variables (`is_viral_hit`, `weeks_on_top_100`).

## Quick Start

```python
import pandas as pd

meta = pd.read_csv("track_metadata.csv")
audio = pd.read_csv("audio_features.csv")
panel = pd.read_csv("chart_longevity_panel.csv")

df = meta.merge(audio, on="track_id").merge(panel, on="track_id")
print(df.head())
```

## Suggested Projects
- Viral Hit Predictor: Train XGBoost/LightGBM to predict is_viral_hit from audio features alone.
- Audio Trend EDA: Visualize how song length and tempo have evolved over time.
- Genre Profiling: Compare energy vs. valence distributions across genres using Seaborn scatter plots.