from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()  # Add this line

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing function
def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())  # Remove non-alphanumeric characters
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['text']
    # Preprocess the input
    cleaned_input = preprocess(user_input)
    # Transform the input using the vectorizer
    input_vector = vectorizer.transform([cleaned_input])
    # Make a prediction
    prediction = model.predict(input_vector)[0]
    # Map prediction to sentiment
    sentiment = "Positive" if prediction == 2 else "Neutral" if prediction == 1 else "Negative"
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == '__main__':
    app.run(debug=True)