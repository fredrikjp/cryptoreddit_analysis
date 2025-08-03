# Use an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /real_time_reddit_sentiment

# Copy all project files into the container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Choose script to run
CMD ["echo", "Specify script via CMD"]

