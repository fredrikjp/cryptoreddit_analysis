from kafka import KafkaConsumer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from openai import OpenAI
import numpy as np
from utils import assert_response_format
from collections import deque


MAX_LEN = 100
score_queues = {}          # subreddit → deque of (time_idx, score)
score_queues_gpt = {}
time_counter = 0
time_counter_gpt = 0

#API key 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
# Initialize Kafka consumer
consumer = KafkaConsumer(
    'reddit_comments',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='sentiment-consumer-group'
)


print("🟢 Listening for messages...")

# Process incoming messages
analyzer = SentimentIntensityAnalyzer()

# Initialize the OpenAI client
client = OpenAI(api_key = "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"
)

# We want to batch comments to save API costs
Comment_batch = []
Comment_list = []
batch_size = 5
i = 0

# OpenAI sentiment analysis prompt
input = f"You are a crypto sentiment analyzer. For any Reddit comment and its subreddit, return 5 float values between 0 and 1:   [bearish, neutral, bullish, compound, relevance] - compound is the overall sentiment score. - relevance should be 1 if the comment is directed at the subreddit subject else how relevant. Output must be a list of floats only Json format."

for message in consumer:
    # Extract subreddit and comment text from the message
    subreddit = message.value.get("subreddit")
    text = message.value.get("text")
    
    # Dict holding "subreddit" and "text" keys
    comment = message.value
    
    # Skip if the comment is empty or whitespace 
    if not text.strip():
        continue

    # Free sentiment analysis using TextBlob, bad performance on understanding context such as irony and general knowledge
    sentiment = analyzer.polarity_scores(text)

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
            val = gpt_score[idx]
            if sb not in score_queues_gpt:
                score_queues_gpt[sb] = deque(maxlen=MAX_LEN)
            score_queues_gpt[sb].append((time_counter_gpt, val))
            time_counter_gpt += 1

        # Send TextBlob data
        val = sentiment
        sb = comment["subreddit"]
        if sb not in score_queues:
            score_queues[sb] = deque(maxlen=MAX_LEN)
        score_queues[sb].append((time_counter, val))
        time_counter += 1

        breakpoint()

    print(f"📝 Comment: {comment}")
    print(f"📈 Sentiment score: {sentiment}")
    print("-" * 50)

    # Print the response
