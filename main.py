from flask import Flask, render_template, request
from joblib import load
import numpy as np
from savedmodels.demo import text_preprocessing_pipeline, updateCSV
from datetime import datetime

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

        monitor_file = {'input': user_text_input,
                        'prediction': str_label[text_prediction],
                        'date-time': date_time
                        }
        updateCSV(monitor_file, 'monitor.csv')

    return render_template("index.html", text_pred="The text has been classified as {}".format(pred_text),
                           top_preds=top_preds)


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8080, debug=True)
    app.run(host='0.0.0.0', port=8080) # to access from outside of docker 

