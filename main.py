from flask import Flask, render_template, request
from joblib import load
import numpy as np
from savedmodels.demo import text_preprocessing_pipeline
from datetime import datetime
import csv
import os
from pymongo import MongoClient

app = Flask(__name__)

vectorizer = load('./savedmodels/tfidf_vectorizer.joblib')
print(vectorizer)
model = load('./savedmodels/xgboost_model.joblib')
print(model)

str_label = {
    0: 'neutral',
    1: 'confusion/curiosity',
    2: 'realization/surprise',
    3: 'sadness/grief',
    4: 'approval/pride',
    5: 'admiration',
    6: 'fear/nervousness/embarrassment',
    7: 'gratitude/relief',
    8: 'joy/amusement',
    9: 'remorse/disappointment',
    10: 'desire/excitement/optimism',
    11: 'annoyance/anger',
    12: 'disapproval/disgust',
    13: 'love/caring'
}

MONGODB_URI = 'mongodb+srv://humzamalikramzan:W06dI5K1TnjIb9Jf@nlp.cvp3aqy.mongodb.net/'
DB_NAME = 'deployment_DB'
COLLECTION_NAME = 'monitor'

# def updateCSV(dict_object, file_name):
#     # Get the absolute path of the file
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(script_dir, file_name)

#     # Write the dictionary to the CSV file
#     columns = ['input', 'prediction', 'date-time']
#     with open(file_path, 'a', newline='') as file:
#         writer = csv.DictWriter(file, fieldnames=columns)
#         writer.writerow(dict_object)
        

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predictor():
    if request.method == 'POST':
        # User enter text
        user_text_input = request.form['text']

        # preprocess text: tokenize, stemm, remove emojis, and stopwords
        processed_text = text_preprocessing_pipeline(user_text_input)
        processed_text_str = ' '.join(processed_text)

        # vectorizer text
        vectorized_text = vectorizer.transform([processed_text_str])

        # predict text using model
        text_prediction = model.predict(vectorized_text)
        text_prediction = int(text_prediction)

        # map label to its name
        pred_text = str_label[text_prediction]

        # calculate probabilty of predicted label
        probs = model.predict_proba(vectorized_text)[0]

        # calculate probabilty of all other top labels
        top_probs = probs.argsort()[-10:][::-1]
        top_labels = [str_label[label] for label in top_probs]
        top_percents = [probs[label] for label in top_probs]
        top_preds = list(zip(top_labels, top_percents))

        date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        client = MongoClient(MONGODB_URI)
        db = client.get_database(DB_NAME)
        collection = db[COLLECTION_NAME]

        data = {
            'input': user_text_input,
            'prediction': pred_text,
            'date-time': date_time
        }
        collection.insert_one(data)

    return render_template("index.html", text_pred="The text has been classified as {}".format(pred_text),
                           top_preds=top_preds)


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8080, debug=True)
    app.run(host='0.0.0.0', port=8080) # to access from outside of docker 

