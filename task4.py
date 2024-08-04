import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Display basic information about the dataset
print(df.head())
print(df.info())

# Clean and preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']

df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment_score)

# Plot distribution of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment_score'], bins=30, kde=True)
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# If timestamp data is available, perform time series analysis
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Resample to daily frequency and calculate mean sentiment score
    daily_sentiment = df['sentiment_score'].resample('D').mean()

    plt.figure(figsize=(12, 6))
    daily_sentiment.plot()
    plt.title('Daily Average Sentiment Score')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.show()

# Plot sentiment score by category if available
if 'category' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='category', y='sentiment_score', data=df)
    plt.title('Sentiment Score by Category')
    plt.xlabel('Category')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=45)
    plt.show()
