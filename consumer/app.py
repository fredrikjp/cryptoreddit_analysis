import streamlit as st
import pandas as pd
import json
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
import wordcloud
import os

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
st.sidebar.header("Filters")
subreddits = st.sidebar.multiselect(
    "Select Subreddits",
    options=sentiment_df['Subreddit'].unique(),
    default=sentiment_df['Subreddit'].unique()
)
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

# Display top 10 comments judged by gpt sentiment
st.header("Top 10 Comments by GPT Sentiment")
gpt_sentiment_df = filtered_sentiment_df[filtered_sentiment_df['Source'] == 'GPT']

# Add sentiment scores to the comments DataFrame
comment_and_sentiment_df = filtered_comments_df.merge(
    gpt_sentiment_df[['ID', 'Compound', 'Positive', 'Negative']],
    on=['ID'],
    how='left'
)
# Sort by compound sentiment score and select top 10

st.dataframe(comment_and_sentiment_df[["Timestamp", "Compound", "Positive", "Negative", "Subreddit", "Comment"]].sort_values(by="Compound", ascending=False) , use_container_width=True)

# Sentiment analysis overview
st.header("Sentiment Analysis")
st.subheader("Average Sentiment Scores by Subreddit")
avg_sentiment = filtered_sentiment_df.groupby(['Subreddit', 'Source'])[['Negative', 'Neutral', 'Positive', 'Compound']].mean().reset_index()

# Bar chart for average sentiment
fig_bar = px.bar(
    avg_sentiment,
    x='Subreddit',
    y=['Negative', 'Neutral', 'Positive'],
    barmode='group',
    color_discrete_map={'Negative': 'red', 'Neutral': 'gray', 'Positive': 'green'},
    title="Average Sentiment Scores by Subreddit",
    facet_col='Source'
)
st.plotly_chart(fig_bar, use_container_width=True)

# Line chart for sentiment over time
st.subheader("Sentiment Over Time")
fig_line = px.line(
    filtered_sentiment_df,
    x='Timestamp',
    y='Compound',
    color='Subreddit',
    line_group='Source',
    title="Compound Sentiment Score Over Time",
    hover_data=['Source']
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
