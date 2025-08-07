from kafka import KafkaConsumer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from openai import OpenAI
import numpy as np
from collections import deque
import os
import time

from utils import assert_response_format
from utils import dump_data

# Get the directory of the current script
base_dir = os.path.dirname(__file__)
gpt_json = os.path.join(base_dir, 'data', 'gpt_sentiment_data.json')
vader_json = os.path.join(base_dir, 'data', 'vader_sentiment_data.json')
comments_json = os.path.join(base_dir, 'data', 'comments_data.json')

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

# Que length
MAX_LEN = 1000

if not os.path.exists(vader_json) or os.path.getsize(vader_json) == 0:
    score_queues_vader = {}
else:
    with open(vader_json, 'r') as f:
        score_queues_vader = {
            key: deque(value, maxlen=MAX_LEN) for key, value in json.load(f).items()
        }

if not os.path.exists(gpt_json) or os.path.getsize(gpt_json) == 0:
    score_queues_gpt = {}
else:
    with open(gpt_json, 'r') as f:
        score_queues_gpt = {
            key: deque(value, maxlen=MAX_LEN) for key, value in json.load(f).items()
        }

if not os.path.exists(comments_json) or os.path.getsize(comments_json) == 0:
    comments_queues = {}
else:
    with open(comments_json, 'r') as f:
        comments_queues = {
            key: deque(value, maxlen=MAX_LEN) for key, value in json.load(f).items()
        }


# Initialize Kafka consumer
consumer = KafkaConsumer(
    'reddit_comments',
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    group_id='sentiment-consumer-group',
    session_timeout_ms=120000,        # default is 10000 ms (10s)
    heartbeat_interval_ms=20000,     # default is 3000 ms
)


print("🟢 Listening for messages...")

# Process incoming messages
analyzer = SentimentIntensityAnalyzer()

api_key = os.environ['OPENAI_API_KEY']
# Initialize the OpenAI client
client = OpenAI(api_key = api_key)

# We want to batch comments to save API costs
Comment_batch = []
Sentiment_batch = []
batch_size = 5
i = 0

# OpenAI sentiment analysis prompt
input = f"You are a crypto sentiment analyzer. For any Reddit comment and its subreddit, return 5 float values between 0 and 1:   [bearish, neutral, bullish, compound, relevance] - compound is the overall sentiment score. - relevance should be 1 if the comment is directed at the subreddit subject else how relevant. Output must be a list of floats only Json format."

for message in consumer:
    # Extract subreddit and comment text from the message
    subreddit = message.value.get("subreddit")
    text = message.value.get("text")
    time_stamp = message.value.get("time")
    
    # Dict holding "time", "subreddit" and "text" keys
    comment = message.value
    
    # Skip if the comment is empty or whitespace 
    if not text.strip():
        continue

    # Free sentiment analysis using TextBlob, bad performance on understanding context such as irony and general knowledge
    sentiment = analyzer.polarity_scores(text)
    Sentiment_batch.append(sentiment)

    Comment_batch.append(comment)
 
    i += 1
    if i % batch_size == 0 and i > 4:
        response_batch = client.responses.create(
            model="gpt-4.1-mini",
            input = input + f" {Comment_batch[-batch_size:]}"
            )
        print(f"🔍 OpenAI Response: {response_batch.output_text}")

        try:
            assert_response_format(response_batch.output_text, value_constraints=np.array([[0, 1], [0, 1], [0, 1], [-1, 1], [0, 1]]))
        except:
            print("❌ Response format assertion failed. Skipping this batch.")
            continue
        
        gpt_analysis = np.array(json.loads(response_batch.output_text))

        # Get gpt scores by multiplying each comment score with relevance score, and then averaging them
        gpt_score = gpt_analysis[:, -1][:, np.newaxis]*gpt_analysis[:,:-1]
        
        # Send gpt data in a thread-safe decoupled manner
        for idx, comment in enumerate(Comment_batch[-batch_size:]):
            sb = comment["subreddit"]
            id = comment["id"]
            val = gpt_score[idx]
            # Convert to dictionary 
            val_dict = dict(zip(["neg", "neu", "pos", "compound"], val))
            if sb not in score_queues_gpt:
                score_queues_gpt[sb] = deque(maxlen=MAX_LEN)
            score_queues_gpt[sb].append([id, val_dict])

            # Send VADER data
            val_dict_vader = Sentiment_batch[idx]
            if sb not in score_queues_vader:
                score_queues_vader[sb] = deque(maxlen=MAX_LEN)
            score_queues_vader[sb].append([id, val_dict_vader])

            if sb not in comments_queues:
                comments_queues[sb] = deque(maxlen=MAX_LEN)
            comments_queues[sb].append([id, time_stamp, comment["text"]])
        dump_data(score_queues_gpt, gpt_json) 
        dump_data(score_queues_vader, vader_json)
        dump_data(comments_queues, comments_json)
        # Clear the batch
        Comment_batch.clear()
        Sentiment_batch.clear()


    # Print the response
    print(f"🔔 Timestamp: {time_stamp}")
    print(f"📝 Comment: {comment['text']}")
    print(f"📈 Sentiment score: {sentiment}")
    print("-" * 50)

    consumer.commit() # Commit the offset after processing the message. Let's the consumer know that the message has been processed such that it won't be reprocessed.
