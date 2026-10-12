from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Global Spotify Hits Dashboard",
    page_icon="🎵",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_data():
    metadata = pd.read_csv(DATA_DIR / "track_metadata.csv")
    audio = pd.read_csv(DATA_DIR / "audio_features.csv")
    performance = pd.read_csv(DATA_DIR / "chart_longevity_panel.csv")

    # Each source file has one row per track; validate the joins to avoid duplication.
    df = metadata.merge(audio, on="track_id", validate="one_to_one")
    df = df.merge(performance, on="track_id", validate="one_to_one")
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["duration_min"] = df["duration_ms"] / 60000
    df["viral_label"] = df["is_viral_hit"].map({1: "Viral hit", 0: "Not marked viral"})
    return df


df = load_data()

st.title("🎵 Global Spotify Hits Dashboard")
st.caption(
    "For a playlist programming manager: compare song longevity and viral-hit rates "
    "to guide weekly playlist review and genre-level promotion decisions."
)

st.sidebar.header("Explore the catalog")
years = sorted(df["release_year"].dropna().astype(int).unique().tolist())
year_range = st.sidebar.select_slider(
    "Release year",
    options=years,
    value=(min(years), max(years)),
)
genres = sorted(df["genre"].dropna().unique().tolist())
selected_genres = st.sidebar.multiselect(
    "Genres for KPI and track details",
    options=genres,
    default=genres,
)
min_weeks = st.sidebar.slider(
    "Minimum weeks on the Top 100",
    min_value=int(df["weeks_on_top_100"].min()),
    max_value=int(df["weeks_on_top_100"].max()),
    value=int(df["weeks_on_top_100"].min()),
)

filtered = df[
    df["release_year"].between(year_range[0], year_range[1])
    & df["genre"].isin(selected_genres)
    & (df["weeks_on_top_100"] >= min_weeks)
].copy()

st.caption(
    f"Release dates represented: {df['release_date'].min():%b %Y}–"
    f"{df['release_date'].max():%b %Y} · Supplied CSV snapshot · "
    "Production refresh target: daily; manager review: weekly."
)

if filtered.empty:
    st.warning("No tracks match these filters. Widen the year range or lower the minimum weeks.")
    st.stop()

# Three headline KPIs: catalog size, typical chart longevity, and viral-hit share.
k1, k2, k3 = st.columns(3)
k1.metric("Tracks in selection", f"{len(filtered):,}")
k2.metric("Median weeks on Top 100", f"{filtered['weeks_on_top_100'].median():.0f}")
k3.metric("Marked viral hits", f"{filtered['is_viral_hit'].mean():.1%}")

st.subheader("What helps songs stay on the chart?")
st.caption(
    "Median weeks on the Top 100 by genre. This benchmark uses the selected release years "
    "and minimum-weeks threshold, and compares all genres so the manager can spot differences."
)
benchmark = df[
    df["release_year"].between(year_range[0], year_range[1])
    & (df["weeks_on_top_100"] >= min_weeks)
]
genre_longevity = (
    benchmark.groupby("genre", as_index=False)
    .agg(
        median_weeks=("weeks_on_top_100", "median"),
        tracks=("track_id", "count"),
    )
    .sort_values("median_weeks", ascending=False)
)
fig1 = px.bar(
    genre_longevity,
    x="genre",
    y="median_weeks",
    hover_data={"tracks": True, "median_weeks": ":.1f"},
    labels={"genre": "Genre", "median_weeks": "Median weeks on Top 100"},
)
fig1.update_layout(margin=dict(l=10, r=10, t=20, b=10), xaxis_tickangle=-25)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Which genres have more songs marked as viral hits?")
genre_viral = (
    benchmark.groupby("genre", as_index=False)
    .agg(
        viral_rate=("is_viral_hit", "mean"),
        tracks=("track_id", "count"),
    )
    .sort_values("viral_rate", ascending=False)
)
genre_viral["viral_rate_pct"] = genre_viral["viral_rate"] * 100
fig2 = px.bar(
    genre_viral,
    x="genre",
    y="viral_rate_pct",
    hover_data={"tracks": True, "viral_rate_pct": ":.1f"},
    labels={"genre": "Genre", "viral_rate_pct": "Tracks marked viral (%)"},
)
fig2.update_layout(margin=dict(l=10, r=10, t=20, b=10), xaxis_tickangle=-25)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Inspect matching tracks"):
    detail_cols = [
        "track_title", "artist_name", "genre", "release_date",
        "weeks_on_top_100", "peak_chart_rank", "total_streams_mil",
        "is_viral_hit",
    ]
    details = filtered[detail_cols].sort_values(
        ["weeks_on_top_100", "total_streams_mil"], ascending=False
    )
    st.dataframe(
        details.rename(
            columns={
                "track_title": "Track",
                "artist_name": "Artist",
                "genre": "Genre",
                "release_date": "Release date",
                "weeks_on_top_100": "Weeks on Top 100",
                "peak_chart_rank": "Best chart rank (1 = best)",
                "total_streams_mil": "Streams (millions)",
                "is_viral_hit": "Marked viral (1 = yes)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "Data note: the supplied files contain one row per track and a release date, not a "
    "weekly time series. Charts describe differences across tracks; they do not show "
    "week-by-week movement. Confirm the original Kaggle dataset page and whether records "
    "are real or synthetic before presenting this as real-world Spotify data. "
    "The 'viral hit' field is a supplied label, not a causal explanation."
)
