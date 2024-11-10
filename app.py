import os
import joblib
import requests
import re
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()

# Retrieve YouTube API key and set port from environment variables
# If you have no API key, create one from YouTube.
# Add a .env file to the root of your project and save the following entries in the .env file:
# Just rename .env.example to .env and enter your own api key
# YOUTUBE_API_KEY='your_actual_youtube_api_key_here'
# PORT=5000

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise EnvironmentError("YOUTUBE_API_KEY is not set. Please add it to your environment variables.")

PORT = int(os.environ.get("PORT", 5000))

# Model paths dictionary
# Preload models and tokenizers at startup for optimized performance
model_dict = {
    'logistic_regression': joblib.load('models/logistic_regression_model.pkl'),
    'xgboost': joblib.load('models/xgboost_model.pkl'),
    # 'random_forest': joblib.load('models/random_forest_model.pkl'),
    'support_vector_machine': joblib.load('models/support_vector_machine_model.pkl'),
    'multinomial_naive_bayes': joblib.load('models/multinomial_naive_bayes_model.pkl')
}

# Load LSTM model and tokenizer only once if memory allows
# lstm_model = load_model('models/lstm_deep_learning_model.h5')
# lstm_tokenizer = joblib.load('models/lstm_deep_learning_tokenizer.pkl')


def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\b\w*@\w*\.\w*\b', '', text)  # Remove email addresses
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtag symbols
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Remove repeated characters (e.g., "soooo" -> "soo")
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)  # Remove consecutive duplicate words with spaces

    # Handle words repeated without spaces, like "trumptrumptrump"
    text = re.sub(r'(\b\w+)\1+', r'\1', text)

    # Final cleanup to reduce consecutive duplicates
    text = re.sub(r'(\b\w+)(\1)+', r'\1', text)  # Reduce repeated words to a single instance

    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text


# Define a function to truncate long words or emoji sequences
def truncate_long_words(comment, max_word_length=45, max_emoji_sequence=5):
    # Emoji pattern to match emojis, considering combining characters and sequences
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U0001F004-\U0001F0CF"  # Playing Cards
        "]+"
    )

    words = comment.split()  # Split the comment into words
    truncated_words = []  # List to store truncated words

    for word in words:
        emojis_in_word = emoji_pattern.findall(word)  # Find emojis in the word

        if emojis_in_word:
            # Extract text and emojis
            text_part = re.sub(emoji_pattern, '', word)  # Remove emojis from word
            emoji_part = ''.join(emojis_in_word)  # Get all emojis from word

            # If emoji sequence is too long, truncate it
            if len(emoji_part) > max_emoji_sequence:
                emoji_part = emoji_part[:max_emoji_sequence] + '...'  # Truncate and add ellipsis

            # Add the word with truncated emojis back into the list
            if text_part:
                truncated_words.append(text_part + emoji_part)  # Keep both text and emojis
            else:
                truncated_words.append(emoji_part)  # If no text, just emojis
        else:
            # If the word doesn't have emojis, just check its length
            truncated_words.append(word if len(word) <= max_word_length else word[:max_word_length] + '...')

    return ' '.join(truncated_words)  # Return the processed comment


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


# Endpoint for UI
@app.route('/')
def index():
    return render_template('frontend.html')


# Endpoint for troubleshooting UI when it is needed.
@app.route('/api')
def home():
    return jsonify(message="Welcome to the Sentiment Analysis API!")


@app.route('/analyze', methods=['POST'])
def analyze():
    input_value = request.json.get('input_value')
    model_choice = request.json.get('model_choice') or 'xgboost'

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

    try:
        if model_choice == 'lstm_deep_learning':
            # Use preloaded LSTM model and tokenizer !! Not possible having memory quota at render.com

            lstm_model = load_model('models/lstm_deep_learning_model.h5')
            lstm_tokenizer = joblib.load('models/lstm_deep_learning_tokenizer.pkl')

            max_sequence_length = 100
            cleaned_comments = [clean_text(comment) for comment in comments]
            sequences = lstm_tokenizer.texts_to_sequences(cleaned_comments)
            padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
            predictions = lstm_model.predict(padded_sequences)
            sentiments = ["Positive" if pred > 0.5 else "Negative" for pred in predictions]
        else:
            # Use preloaded traditional ML model and vectorizer
            model = model_dict[model_choice]
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

    # Let us shorten words > 45 characters and emoji > 5 characters for better UI display
    comments = [truncate_long_words(comment) for comment in comments]

    # Limit how many comments are displayed to prevent page bloating
    result = {
        'summary': summary,
        'comments': comments[:500],
        'sentiments': sentiments[:500]
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
