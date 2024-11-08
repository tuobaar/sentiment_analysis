import os
import joblib
import requests
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Retrieve API key and set port from environment variables
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
PORT = int(os.environ.get("PORT", 5000))

# Model paths dictionary
model_dict = {
    'logistic_regression': 'models/logistic_regression_model.pkl',
    'xgboost': 'models/xgboost_model.pkl',
    'random_forest': 'models/random_forest_model.pkl',
    'support_vector_machine': 'models/support_vector_machine_model.pkl',
    'multinomial_naive_bayes': 'models/multinomial_naive_bayes_model.pkl',
    'lstm_deep_learning': 'models/lstm_deep_learning_model.h5'
}

# Tokenizer paths dictionary
tokenizer_dict = {
    'lstm_deep_learning': 'models/lstm_deep_learning_tokenizer.pkl'
}


def clean_text(text):
    # All the regex cleaning functions as you had before
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\b\w*@\w*\.\w*\b', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def truncate_long_words(comment, max_word_length=45, max_emoji_sequence=5):
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
        "\U0001F004-\U0001F0CF]+")
    words = comment.split()
    truncated_words = []

    for word in words:
        emojis_in_word = emoji_pattern.findall(word)
        if emojis_in_word:
            text_part = re.sub(emoji_pattern, '', word)
            emoji_part = ''.join(emojis_in_word)
            if len(emoji_part) > max_emoji_sequence:
                emoji_part = emoji_part[:max_emoji_sequence] + '...'
            truncated_words.append(text_part + emoji_part if text_part else emoji_part)
        else:
            truncated_words.append(word if len(word) <= max_word_length else word[:max_word_length] + '...')

    return ' '.join(truncated_words)


def extract_video_id(url):
    match = re.match(
        r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]+)', url)
    return match.group(1) if match else url


def get_youtube_comments(video_url, api_key):
    video_id = extract_video_id(video_url)
    comments, next_page_token = [], None

    while True:
        url = (f"https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&textFormat"
               f"=plainText&maxResults=100&key={api_key}")
        if next_page_token:
            url += f"&pageToken={next_page_token}"

        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        data = response.json()
        comments += [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in data.get('items', [])]
        next_page_token = data.get("nextPageToken")

        if not next_page_token:
            break

    return comments


@app.route('/analyze', methods=['POST'])
def analyze():
    input_value = request.json.get('input_value')
    model_choice = request.json.get('model_choice')

    if not model_choice:
        return jsonify({
            'error': 'No model selected. Would you like to analyze with a default model (XGBoost)?',
            'confirmation_required': True
        })

    if re.match(r'^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$', input_value):
        comments = get_youtube_comments(input_value, YOUTUBE_API_KEY)
        if not comments:
            return jsonify({'error': 'No comments found or an error occurred retrieving comments.'})
    else:
        comments = [input_value]

    model_choice = model_choice or 'xgboost'
    model_path = model_dict.get(model_choice)

    try:
        if model_choice == 'lstm_deep_learning':
            model = load_model(model_path)
            tokenizer = joblib.load(tokenizer_dict[model_choice])
            max_sequence_length = 100

            cleaned_comments = [clean_text(comment) for comment in comments]
            sequences = tokenizer.texts_to_sequences(cleaned_comments)
            padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

            predictions = model.predict(padded_sequences)
            sentiments = ["Positive" if pred > 0.5 else "Negative" for pred in predictions]
        else:
            model = joblib.load(model_path)
            vectorizer = joblib.load(f'vectorizers/{model_choice}_tfidf_vectorizer.pkl')

            cleaned_comments = [clean_text(comment) for comment in comments]
            comments_vectorized = vectorizer.transform(cleaned_comments)
            predictions = model.predict(comments_vectorized)
            sentiments = ["Positive" if pred == 1 else "Negative" for pred in predictions]

    except Exception as e:
        return jsonify({'error': f"Error loading model or processing data: {e}"})

    positive = sum(sent == "Positive" for sent in sentiments)
    negative = len(sentiments) - positive
    summary = {
        "positive": positive,
        "negative": negative,
        "num_comments": len(comments),
        "rating": (positive / len(comments)) * 100
    }

    comments = [truncate_long_words(comment) for comment in comments]

    if len(comments) > 500:
        result = {
            'summary': summary,
            'comments': comments[:500],
            'sentiments': sentiments[:500]
        }
    else:
        result = {
            'summary': summary,
            'comments': comments,
            'sentiments': sentiments
        }
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
