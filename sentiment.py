import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from afinn import Afinn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string
import os
import googleapiclient.discovery
from nltk.tokenize import PunktTokenizer
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for servers or CLI

#nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Setup YouTube API
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = "AIzaSyCKoAKcO3b_6ztkc5MH-zxc_V-Kd2AR9Sw"  # Replace with your actual API key

def fetch_youtube_comments(video_id):
    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)
    request = youtube.commentThreads().list(
        part="id,snippet", videoId=video_id, maxResults=100, order="relevance"
    )
    response = request.execute()

    comments = []
    for item in response['items']:
        comment_data = {
            'comment': item['snippet']['topLevelComment']['snippet']['textDisplay'],
            'like_count': item['snippet']['topLevelComment']['snippet']['likeCount']
        }
        comments.append(comment_data)

    comments.sort(key=lambda x: x['like_count'], reverse=True)
    return pd.DataFrame(comments[:100])

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def calculate_sentiment_score(text):
    afinn = Afinn()
    return afinn.score(text)

def calculate_sentiment_impact(row):
    if row['like_count'] == 0:
        return row['Sentiment_Score']
    else:
        return row['Sentiment_Score'] * row['like_count']

def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'



def generate_pie_chart(positive, neutral, negative):
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [positive, neutral, negative]
    
    # Updated colors
    colors = ['#28a745', '#007bff', '#dc3545']  # Green, Blue, Red
    explode = (0.05, 0.05, 0.05)

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        shadow=True
    )
    plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.

    # Ensure output directory exists
    output_dir = 'static'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'pie_chart.png')
    plt.savefig(output_path)
    plt.close()


def analyze_sentiment(video_id):
    df = fetch_youtube_comments(video_id)
    if df.empty:
        return {"error": "No comments found"}

    df['processed_text'] = df['comment'].apply(preprocess_text)
    df['Sentiment_Score'] = df['comment'].apply(calculate_sentiment_score)
    df['Sentiment_Impact'] = df.apply(calculate_sentiment_impact, axis=1)
    df['category'] = df['Sentiment_Score'].apply(categorize_sentiment)

    # Train a Naive Bayes model for educational purpose (optional)
    X = df['processed_text']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    # Count percentages
    sentiment_counts = pd.Series(y_pred).value_counts(normalize=True) * 100
    positive = round(sentiment_counts.get('Positive', 0.0), 2)
    neutral = round(sentiment_counts.get('Neutral', 0.0), 2)
    negative = round(sentiment_counts.get('Negative', 0.0), 2)

    # Save pie chart
    generate_pie_chart(positive, neutral, negative)
    #plt.savefig('static/pie_chart.png')

    # Return sentiment result
    return {
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
        "chart": "static/pie_chart.png"
    }

# For testing locally
if __name__ == "__main__":
    result = analyze_sentiment("dyrETLlrrC4")  # Replace with any real video ID
    print(result)
