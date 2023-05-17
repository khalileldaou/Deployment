from joblib import load
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re


vectorizer = load('./savedmodel/tfidf_vectorizer.joblib')
print(vectorizer)
model = load('./savedmodel/xgboost_model.joblib')
print(model)

# Download the required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define the preprocessing functions
def remove_emojis(text):
    # Remove emojis from text using regex
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def tokenize(text):
    return word_tokenize(text.lower())

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [t for t in tokens if not t in stop_words]

def stem(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

# Map labels to their label names
labels_name = {
    0: 'admiration',
    1: 'confusion/curiosity',
    2: 'realisation/surprise',
    3: 'sadness/grief',
    4: 'approval/pride',
    5: 'fear/nervousness/embarrassment',
    6: 'gratitude/relief',
    7: 'joy/amusement',
    8: 'remorse/disappointment',
    9: 'desire/excitement/optimism',
    10: 'disgust/disapproval',
    11: 'anger/annoyance',
    12: 'love/caring',    
    13: 'neutral'}


while(True):
    user_text_input = input("Enter Text: ")

    # Apply the preprocessing functions to the text data
    input_tokens = stem(remove_stopwords(tokenize(remove_emojis(user_text_input))))
    print(input_tokens)

    # Concatenate the tokens into a single string
    joined_tokens = ' '.join(input_tokens)
    # Transform the input text using the vectorizer
    vectorized_text = vectorizer.transform([joined_tokens])
    print(vectorized_text)

    # run model
    text_prediction = model.predict(vectorized_text)
    print(text_prediction, labels_name[text_prediction[0]])

    