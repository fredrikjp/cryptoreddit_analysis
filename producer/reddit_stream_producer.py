from kafka import KafkaProducer
import json
import praw
from datetime import datetime
import time

# === Reddit API credentials ===
reddit = praw.Reddit(
    client_id="RKMMnIPxdJxHw8oPLruBDw",
    client_secret="LXAXBwRbsNnFYXpGOjQ7vQGYn0mNIQ",
    user_agent="sentiment-app by u/Thin-Hunter-3593"
)

# === Kafka setup ===
producer = KafkaProducer(
    bootstrap_servers="kafka:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# === Subreddits to monitor ===
subreddits = reddit.subreddit("CryptoCurrency+Bitcoin+btc+Ethereum+eth+Dogecoin+doge+Solana+solana")
print("🚀 Listening to new comments on Reddit...")


for idx, comment in enumerate(subreddits.stream.comments(skip_existing=True)):
    if True:#comment.distinguished == "moderator" or comment.distinguished == "admin" or comment.distinguished == "special":
        timestamp = datetime.utcfromtimestamp(comment.created_utc).isoformat()
        sub = str(comment.subreddit)
        text = comment.body
        id = comment.id
        data = {"time": timestamp, "subreddit": sub, "text": text, "id": id}
        producer.send("reddit_comments", value=data)
        print(f"Comment from {sub}: {text[:80]}...")
