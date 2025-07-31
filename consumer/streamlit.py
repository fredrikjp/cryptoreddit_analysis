import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
import time
import os
import json
import numpy as np
import typing

# --- Load sentiment data from JSON dump created by consumer ---
GPT_FILE = "gpt_sentiment_data.json"
VADER_FILE = "vader_sentiment_data.json"

# Function to load JSON safely
def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

score_queues_gpt = load_json(GPT_FILE)
score_queues_vader = load_json(VADER_FILE)

if not score_queues_gpt and not score_queues_vader:
    st.warning("No sentiment data available in either file.")
    st.stop()

st.set_page_config(page_title="Sentiment Candlestick Viewer", layout="wide")
st.title("📈 GPT-Based Sentiment Candlestick Visualization")

# Select the sentiment data to visualize
selected_subreddit = st.selectbox("Choose a subreddit to display:", sorted(score_queues_gpt.keys()))
selected_score = st.selectbox("Choose which score to display:", ["pos", "neg", "neu", "compound"])
candle_interval = st.selectbox("Time bucket size:", ["15s", "30s", "1Min", "5Min", "1H", "1D"], index=2)

# --- Plotting helper ---
def make_candlestick(data: typing.List[typing.List[typing.Union[str, typing.Dict[str, float]]]], label: str, selected_score) -> go.Candlestick:

    # Map for resampling intervals
    interval_map = {
        "15s": "15s",
        "30s": "30s",
        "1Min": "1min",
        "5Min": "5t",
        "1H": "1h",
        "1D": "1D"
    }

    df = pd.DataFrame(
        [(datetime.fromisoformat(t), s) for t, s in data],
        columns=["timestamp", "sentiment"]
    )
    df.set_index("timestamp", inplace=True)
    df["value"] = df["sentiment"].apply(lambda x: float(x[selected_score])) 
    ohlc = df["value"].resample(interval_map[candle_interval]).ohlc().dropna()

    trace = go.Candlestick(
        x=ohlc.index,
        open=ohlc["open"],
        high=ohlc["high"],
        low=ohlc["low"],
        close=ohlc["close"],
        name=label,
        increasing_line_color='green',
        decreasing_line_color='red'
    )
    return trace

# --- Create and render plot ---
fig = go.Figure()
if selected_subreddit in score_queues_gpt:
    fig.add_trace(make_candlestick(score_queues_gpt[selected_subreddit], "GPT Sentiment", selected_score))
if selected_subreddit in score_queues_vader:
    fig.add_trace(make_candlestick(score_queues_vader[selected_subreddit], "VADER Sentiment", selected_score))


fig.update_layout(
    title=f"Compound Sentiment Candlestick for r/{selected_subreddit}",
    xaxis_title="Time",
    yaxis_title="Compound Sentiment",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)
st.write(pd.DataFrame(score_queues_gpt[selected_subreddit] if selected_subreddit in score_queues_gpt else score_queues_vader[selected_subreddit], columns=["timestamp", "sentiment"]))
# Refresh button
if st.button("🔄 Refresh Now"):
    st.rerun()

# Optional auto-refresh
st.write("Refreshing in 5 seconds...")
time.sleep(5)
st.rerun()
