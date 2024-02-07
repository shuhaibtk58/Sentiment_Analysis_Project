from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import nltk
from tensorflow.keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
from tensorflow.keras.datasets import imdb

nltk.download('punkt')

app = Flask(__name__)

# Load the pre-trained sentiment analysis model
model = load_model('sentiment_analysis_model.h5')

# Function to preprocess the input text
def preprocess_text(text):
    word2id = imdb.get_word_index()
    id2word = {i: word for word, i in word2id.items()}
    tokens = word_tokenize(text.lower())
    word_ids = [word2id.get(token, 0) for token in tokens]
    padded_sequence = sequence.pad_sequences([word_ids], maxlen=500)
    return padded_sequence

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle sentiment analysis
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = preprocess_text(text)
        prediction = model.predict(np.array(processed_text))
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        return render_template('result.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=False)
