import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

# vectorizer = load('./savedmodel/tfidf_vectorizer.joblib')
# print(vectorizer)
# model = load('./savedmodel/xgboost_model.joblib')
# print(model)

# Download the required NLTK data
nltk.download("punkt")
nltk.download("stopwords")


# Define the preprocessing functions
def remove_emojis(text):
    # Remove emojis from text using regex
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def tokenize(text):
    return word_tokenize(text.lower())


def remove_stopwords(tokens):
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if not t in stop_words]


def stem(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def text_preprocessing_pipeline(text):
    # Apply each step of the pipeline sequentially
    text = remove_emojis(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return tokens


#
# def updateCSV(dict_object, file_name):
#     # Write the dictionary to the CSV file
#     columns = ['input', 'prediction', 'date-time']
#     with open(file_name, 'a', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=columns)
#         writer.writerow(dict_object)
#
#
# # Map labels to their label names
# labels_name = {
#     0: 'admiration',
#     1: 'confusion/curiosity',
#     2: 'realisation/surprise',
#     3: 'sadness/grief',
#     4: 'approval/pride',
#     5: 'fear/nervousness/embarrassment',
#     6: 'gratitude/relief',
#     7: 'joy/amusement',
#     8: 'remorse/disappointment',
#     9: 'desire/excitement/optimism',
#     10: 'disgust/disapproval',
#     11: 'anger/annoyance',
#     12: 'love/caring',
#     13: 'neutral'}
#
# while (True):
#     user_text_input = input("Enter Text: ")
#
#     # Apply the preprocessing functions to the text data
#     input_tokens = stem(remove_stopwords(tokenize(remove_emojis(user_text_input))))
#     print(input_tokens)
#
#     # Concatenate the tokens into a single string
#     joined_tokens = ' '.join(input_tokens)
#     # Transform the input text using the vectorizer
#     vectorized_text = vectorizer.transform([joined_tokens])
#     print(vectorized_text)
#
#     # run model
#     text_prediction = model.predict(vectorized_text)
#     print(text_prediction, labels_name[text_prediction[0]])
#
#     # capture time and date dd/mm/YY H:M:S
#     date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
#     print("date-time =", date_time)
#
#     # monitor: save input, prediction, and time-date in a json file
#     monitor_file = {'input': user_text_input, 'prediction': labels_name[text_prediction[0]], 'date-time': date_time}
#     updateCSV(monitor_file, 'monitor.csv')
