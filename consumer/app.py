import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import wordcloud
import os
from datetime import timedelta
import time
import sys, os
import numpy as np

sys.path.append(os.path.dirname(__file__))  # add current dir to sys.path



# Set page configuration
st.set_page_config(page_title="Reddit Comment Analysis", layout="wide")


# Title and description
st.title("Reddit Comment Analysis Dashboard")
st.write("This dashboard displays sentiment analysis of Reddit comments from various subreddits.")





# Load JSON data
def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return {}

# Load data
base_dir = os.path.dirname(__file__)
gpt_json = os.path.join(base_dir, 'data', 'gpt_sentiment_data.json')
vader_json = os.path.join(base_dir, 'data', 'vader_sentiment_data.json')
comments_json = os.path.join(base_dir, 'data', 'comments_data.json')

gpt_sentiment_data = load_json(gpt_json)
vader_sentiment_data = load_json(vader_json)
comments_data = load_json(comments_json)


# Convert timestamp strings to datetime
def parse_timestamp(timestamp):
    try:
        if isinstance(timestamp, (int, float)):
            return pd.to_datetime(timestamp, unit='s')
        return pd.to_datetime(timestamp)
    except:
        return pd.NaT

# Process comments data
def process_comments_data(data):
    rows = []
    for subreddit, comments in data.items():
        for comment in comments:
            id, timestamp, text = comment
            rows.append({
                'Subreddit': subreddit,
                'ID': id,
                'Timestamp': parse_timestamp(timestamp),
                'Comment': text
            })
    return pd.DataFrame(rows)

# Process sentiment data
def process_sentiment_data(data, source):
    rows = []
    for subreddit, entries in data.items():
        for entry in entries:
            id, sentiment = entry
            row = {
                'Subreddit': subreddit,
                'ID': id,
                'Source': source,
                'Negative': sentiment['neg'],
                'Neutral': sentiment['neu'],
                'Positive': sentiment['pos'],
                'Compound': sentiment['compound']
            }
            rows.append(row)
    return pd.DataFrame(rows)

##########################################################################

# Create DataFrames
comments_df = process_comments_data(comments_data)
gpt_sentiment_df = process_sentiment_data(gpt_sentiment_data, 'GPT')
vader_sentiment_df = process_sentiment_data(vader_sentiment_data, 'VADER')

# Combine sentiment data
sentiment_df = pd.concat([gpt_sentiment_df, vader_sentiment_df], ignore_index=True)


# Add timestamp column to sentiment_df by message ID
sentiment_df = sentiment_df.merge(
    comments_df[['ID', 'Timestamp']],
    on=['ID'],
    how='left'
    )


# Sidebar for filtering
st.sidebar.header("Filter Subreddits")
all_subreddits = sorted(sentiment_df['Subreddit'].unique(), key=lambda s: s.lower())
select_all = st.sidebar.checkbox("Select All Subreddits", value=False)

if select_all:
    subreddits = st.sidebar.multiselect(
        "Select Subreddits",
        options=all_subreddits,
        default=list(all_subreddits)
    )
else:
    subreddits = st.sidebar.multiselect(
        "Select Subreddits",
        options=all_subreddits,
        default=["Bitcoin", "ethereum", "solana"]
    )


# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(sentiment_df['Timestamp'].min(), sentiment_df['Timestamp'].max())
)

# Filter data
filtered_comments_df = comments_df[
    (comments_df['Subreddit'].isin(subreddits)) &
    (comments_df['Timestamp'].dt.date >= date_range[0]) &
    (comments_df['Timestamp'].dt.date <= date_range[1])
]
filtered_sentiment_df = sentiment_df[
    (sentiment_df['Subreddit'].isin(subreddits)) &
    (sentiment_df['Timestamp'].dt.date >= date_range[0]) &
    (sentiment_df['Timestamp'].dt.date <= date_range[1])
]

####################################################################

# Display top 10 comments judged by gpt sentiment
st.header("Comments with GPT Scores")
gpt_sentiment_df = filtered_sentiment_df[filtered_sentiment_df['Source'] == 'GPT']
# Add sentiment scores to the comments DataFrame
comment_and_sentiment_df = filtered_comments_df.merge(
    gpt_sentiment_df[['ID', 'Compound', 'Positive', 'Negative']],
    on=['ID'],
    how='left'
)

st.dataframe(comment_and_sentiment_df[["Timestamp", "Compound", "Positive", "Negative", "Subreddit", "Comment"]].sort_values(by="Compound", ascending=False) , use_container_width=True)

##############################################################

# Sentiment analysis overview
st.header("Sentiment Analysis")
st.subheader("Average Sentiment Scores by Subreddit")

# Bar chart time range slider
min_date_bar = filtered_sentiment_df['Timestamp'].min().to_pydatetime()
max_date_bar = filtered_sentiment_df['Timestamp'].max().to_pydatetime()

# Initialize session state for bar chart date range
if 'bar_date_range' not in st.session_state:
    st.session_state.bar_date_range = (min_date_bar, max_date_bar)

start_dt, end_dt = st.session_state.bar_date_range

filtered_bar_df = filtered_sentiment_df[
    (filtered_sentiment_df['Timestamp'] >= start_dt) &
    (filtered_sentiment_df['Timestamp'] <= end_dt)
]

avg_sentiment = filtered_bar_df.groupby(['Subreddit', 'Source'])[['Negative', 'Neutral', 'Positive', 'Compound']].mean().reset_index()

############################################################################

# Animated bar chart
# Add timestamp column to sentiment_df by message ID
gpt_sentiment_df.merge(
    comments_df[['ID', 'Timestamp']],
    on=['ID'],
    how='left'
    )

# --- Aggregation period menu ---
freq_map = {
    "1 Hour": "1h",
    "1 Day": "1D",
    "1 Week": "1W"
}
selected_freq_label = st.selectbox("Aggregation period", list(freq_map.keys()), index=1)
selected_freq = freq_map[selected_freq_label]

# --- Aggregate raw data ---
df_grouped = (
    gpt_sentiment_df
    .groupby([
        pd.Grouper(key='Timestamp', freq=selected_freq),
        'Subreddit',
        'Source'
    ])
    [['Negative', 'Neutral', 'Positive', 'Compound']]
    .mean()
    .reset_index()
)

# Rolling average for compound if 1 hour and 1 day selected
if selected_freq == '1h':
    df_grouped['Compound'] = df_grouped.groupby(['Subreddit', 'Source'])['Compound'].transform(
        lambda x: x.rolling(window=9, min_periods=1).mean()
    )
elif selected_freq == '1D':
    df_grouped['Positive'] = df_grouped.groupby(['Subreddit', 'Source'])['Positive'].transform(
        lambda x: x.rolling(window=2, min_periods=1).mean()
    )
    df_grouped['Negative'] = df_grouped.groupby(['Subreddit', 'Source'])['Negative'].transform(
        lambda x: x.rolling(window=2, min_periods=1).mean()
    )
    df_grouped['Neutral'] = df_grouped.groupby(['Subreddit', 'Source'])['Neutral'].transform(
        lambda x: x.rolling(window=2, min_periods=1).mean()
    )
    df_grouped['Compound'] = df_grouped.groupby(['Subreddit', 'Source'])['Compound'].transform(
        lambda x: x.rolling(window=2, min_periods=1).mean()
    )


# Get all combinations of time x subreddit x source
full_index = pd.MultiIndex.from_product(
    [df_grouped['Timestamp'].unique(),
     df_grouped['Subreddit'].unique(),
     df_grouped['Source'].unique()],
    names=['Timestamp','Subreddit','Source']
)

df_grouped = df_grouped.set_index(['Timestamp','Subreddit','Source']).reindex(full_index, fill_value=None).reset_index()

# --- create Compound Neg and mask negatives from Compound ---
df_plot = df_grouped.copy()

df_plot["Compound Neg"] = np.where(df_plot["Compound"] < 0, -df_plot["Compound"], np.nan)
df_plot["Compound"]     = np.where(df_plot["Compound"] < 0, np.nan, df_plot["Compound"])

df_plot = df_plot.sort_values(["Timestamp", "Subreddit", "Source"]).reset_index(drop=True)



# Make the bars value stay the same for each time period it does not have any data
# Print scores of Btc subreddit for debugging
for timestamp in df_plot['Timestamp'].unique():
    for subreddit in df_plot['Subreddit'].unique():
        # If value is None, fill it with the last available value
        if df_plot.loc[(df_plot['Timestamp'] == timestamp) & (df_plot['Subreddit'] == subreddit), 'Negative'].isnull().any():
            # Get the last available scores for this subreddit
            last_scores = df_plot[
                (df_plot['Subreddit'] == subreddit) &
                (df_plot['Timestamp'] < timestamp)
            ].tail(1)
            if not last_scores.empty:
                df_plot.loc[
                    (df_plot['Timestamp'] == timestamp) & (df_plot['Subreddit'] == subreddit),
                    ['Negative', 'Neutral', 'Positive', 'Compound', 'Compound Neg']
                ] = last_scores[['Negative', 'Neutral', 'Positive', 'Compound', 'Compound Neg']].values[0]

bar_sentiment_melted = df_plot.melt(
    id_vars=['Timestamp', 'Subreddit', 'Source'],
    value_vars=['Negative', 'Neutral', 'Positive', 'Compound', 'Compound Neg'],
    var_name='SentimentType',
    value_name='Score'
)

# (optional) consistent legend/order
order = ["Negative", "Neutral", "Positive", "Compound", "Compound Neg"]
bar_sentiment_melted["SentimentType"] = pd.Categorical(bar_sentiment_melted["SentimentType"], categories=order, ordered=True)





fig_bar = px.bar(
    bar_sentiment_melted,
    x='Subreddit',
    y='Score',  # melted version of Negative/Neutral/Positive
    color='SentimentType',
    animation_frame='Timestamp',  # this is the time axis for animation
    barmode='group',
    color_discrete_map={
        'Negative': 'red',
        'Neutral': 'gray',
        'Positive': 'green',
        'Compound': 'blue'
        ,'Compound Neg': 'black'
    },
    title="Average Sentiment Scores by Subreddit Over Time",
    facet_col='Source'
)

# Optional: adjust animation speed
fig_bar.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800  # ms per frame
fig_bar.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500

st.plotly_chart(fig_bar, use_container_width=True)

#################################################################################

# Animated bubble chart

# --- Add Volume (count of posts/messages) ---
df_grouped["Volume"] = df_grouped.groupby(["Timestamp","Subreddit","Source"])["Positive"].transform("count")

# --- Prepare data for bubble chart ---
df_bubble = df_grouped.copy()
df_bubble["SentimentX"] = (df_bubble["Positive"] - df_bubble["Negative"]) / 2

# --- Static line traces (trails) ---
fig = go.Figure()

for (sub, src), subdf in df_bubble.groupby(["Subreddit","Source"]):
    fig.add_trace(go.Scatter(
        x=subdf["SentimentX"],
        y=subdf["Compound"],
        mode="lines",
        line=dict(width=1),
        name=f"{sub} ({src})",
        showlegend=False,
        hoverinfo="skip"  # don’t spam hover with the trail
    ))


fig_bubble = px.scatter(
    df_bubble,
    x="SentimentX",
    y="Compound",
    size="Volume",              
    color="Subreddit",         
    animation_frame="Timestamp",
    facet_col="Source",
    hover_name="Subreddit",
    title="Sentiment Bubble Chart Over Time",
    size_max=60,              
    range_x=[-0.5, 0.5],
    range_y=[df_bubble["Compound"].min()*1.1, df_bubble["Compound"].max()*1.1]
)

# Optional: match animation speed with bar chart
fig_bubble.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 800
fig_bubble.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500

st.plotly_chart(fig_bubble, use_container_width=True)


# Bar chart for average sentiment
fig_bar = px.bar(
    avg_sentiment,
    x='Subreddit',
    y=['Negative', 'Neutral', 'Positive', 'Compound'],
    barmode='group',
    color_discrete_map={'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green', 'Compound': 'blue'},
    title="Average Sentiment Scores by Subreddit",
    facet_col='Source'
)
st.plotly_chart(fig_bar, use_container_width=True)


#################################################################################


# Line chart for compound sentiment over time usin rolling average
st.subheader("Sentiment Over Time")


# We want the number of points N in the rolling window for each subreddit to be proportional to the subreddit volume.
base_N = 20  # Base number of points for rolling average
volumes = filtered_sentiment_df['Subreddit'].value_counts().to_dict()
median_volume = pd.Series(volumes).median()

rolling_results = []
for sub, scores in gpt_sentiment_df.groupby('Subreddit'):
    N_scaled = max(5, round(base_N * (volumes[sub] / median_volume)))
    scores = scores.copy()
    scores['Rolling_N'] = N_scaled
    scores['Rolling_Compound'] = (
    scores['Compound'].rolling(window=N_scaled, min_periods=N_scaled).mean()
    )
    rolling_results.append(scores)

rolling_df = pd.concat(rolling_results).sort_values(['Subreddit', 'Timestamp'])

fig_line = px.line(
    rolling_df,
    x='Timestamp',
    y='Rolling_Compound',
    color='Subreddit',
    title="Compound Sentiment Score Over Time",
    hover_data=['Subreddit', 'Timestamp', 'Rolling_Compound', 'Rolling_N', 'Source'],
)
st.plotly_chart(fig_line, use_container_width=True)


# Sentiment distribution
st.subheader("Sentiment Distribution")
fig_hist = px.histogram(
    filtered_sentiment_df,
    x='Compound',
    color='Subreddit',
    facet_col='Source',
    nbins=20,
    title="Distribution of Compound Sentiment Scores"
)
st.plotly_chart(fig_hist, use_container_width=True)

# Summary statistics
st.header("Summary Statistics")
st.write("Summary of sentiment scores by subreddit and source.")
st.dataframe(avg_sentiment, use_container_width=True)

# Word cloud (optional, requires wordcloud package)
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    st.header("Word Cloud of Comments")
    selected_subreddit = st.selectbox("Select Subreddit for Word Cloud", subreddits)
    text = ' '.join(filtered_comments_df[filtered_comments_df['Subreddit'] == selected_subreddit]['Comment'].dropna())
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.write("No comments available for word cloud.")
except ImportError:
    st.write("WordCloud package not installed. Skipping word cloud visualization.")

# Additional metrics suggestions (displayed as text)
st.header("Suggested Additional Metrics")
st.markdown("""
To enhance your analysis, consider including the following metrics:
1. **Comment Volume**: Track the number of comments per subreddit over time to identify trends in engagement.
2. **Sentiment Volatility**: Calculate the standard deviation of compound sentiment scores to measure sentiment consistency.
3. **Keyword Frequency**: Analyze the frequency of specific keywords (e.g., 'bull', 'bear', 'scam') to gauge discussion topics.
4. **User Engagement**: If available, include metrics like upvotes, downvotes, or comment replies to assess interaction levels.
5. **Sentiment Correlation**: Compare GPT and VADER sentiment scores to evaluate consistency between models.
6. **Temporal Trends**: Aggregate sentiment by hour/day to detect patterns (e.g., sentiment spikes during market events).
""")
