from flask import Flask, render_template, request
from joblib import load
import json
from savedmodels.demo import text_preprocessing_pipeline
from datetime import datetime
import pymongo
from pymongo import MongoClient

app = Flask(__name__)

# Load vectorzier and xgboost saved models to use for predicting labels for texts
vectorizer = load('./savedmodels/tfidf_vectorizer.joblib')
print(vectorizer)
model = load('./savedmodels/xgboost_model.joblib')
print(model)

# Map labels numbers to their names to show when text is predicted
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

# Link to MongoDB to store the montitoring data
MONGODB_URI = 'mongodb+srv://humzamalikramzan:W06dI5K1TnjIb9Jf@nlp.cvp3aqy.mongodb.net/'
DB_NAME = 'deployment_DB'
COLLECTION_NAME = 'monitor'

# Read data from MongoDB to display the monitoring data in the web page 
def read_data_from_mongodb():
    client = MongoClient(MONGODB_URI)
    db = client.get_database(DB_NAME)
    collection = db[COLLECTION_NAME]
    
    data = collection.find().sort("date_time", pymongo.DESCENDING)
    result = []
    for document in data:
        result.append(document)
    print(result)
    client.close()
    return result
     
# First phase of the app will ask user for text to predict an show the histroy of previous data inputs
@app.route('/')
def home():
    data = read_data_from_mongodb()
    return render_template('index.html', data=data)

# Second phase of the app will show the prediction results
@app.route('/predict', methods=['POST'])
def predictor():
    if request.method == 'POST':
        # if request is a json object, it will return a json object for the results 
        if request.is_json:
            # text is a json object from the notebook
            sentence = request.get_json()['sentence']

            # preprocess text: tokenize, stem, remove emojis, and stopwords
            processed_text = text_preprocessing_pipeline(sentence)
            processed_text_str = ' '.join(processed_text)
            
            # vectorize text with the presaved vectorizer model
            vectorized_text = vectorizer.transform([processed_text_str])

            # predict text using XGBoost model
            text_prediction = model.predict(vectorized_text)
            text_prediction = int(text_prediction)

            # map label to its name
            pred_text = str_label[text_prediction]

            prediction = {'prediction': pred_text}
            return json.dumps(prediction)
        
        # else if the input is a string from the html form
        else: 
            # user enter text
            user_text_input = request.form['text']

            # preprocess text: tokenize, stem, remove emojis, and stopwords
            processed_text = text_preprocessing_pipeline(user_text_input)
            processed_text_str = ' '.join(processed_text)

            # vectorize text with the presaved vectorizer model
            vectorized_text = vectorizer.transform([processed_text_str])

            # predict text using XGBoost model
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

            # get the date and time of predicting this text to add to MongoDB
            date_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

            # save the input text, prediction result, and date-time in MongoDB
            client = MongoClient(MONGODB_URI)
            db = client.get_database(DB_NAME)
            collection = db[COLLECTION_NAME]
            data = {
                'input': user_text_input,
                'prediction': pred_text,
                'date_time': date_time
            }
            collection.insert_one(data)

            # Update the displayed data
            updated_data = read_data_from_mongodb()
            client.close()

            return render_template("index.html", text_pred="The text has been classified as {}".format(pred_text),
                                top_preds=top_preds, data=updated_data)


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8080, debug=True)
    app.run(host='0.0.0.0', port=8080, debug=True) # to access from outside of docker 

