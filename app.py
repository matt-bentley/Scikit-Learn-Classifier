#!flask/bin/python
from flask import Flask
from flask import request
import pickle
from sklearn.externals import joblib 

app = Flask(__name__)

loaded_model = joblib.load('./text_clf_svm_model.pkl')

@app.route('/isAlive')
def index():
    return "true"

@app.route('/predict_title', methods=['POST'])
def get_prediction():
    data = request.json
    titles = []
    for row in data:
        titles.append(row['title'])
    prediction = loaded_model.predict(titles)
    return str(prediction)
   
if __name__ == '__main__':
    app.run()        
    # app.run(debug=True)