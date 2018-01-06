# Scikit-Learn-Classifier

Naive Bayes and SVM classifier examples using the Scikit-Learn library. This project uses the Hacker News dataset for classification however any dataset could be used and optimised using the Grid Search in the project.

## Training

The model can be trained by running model.py. This will use the train.csv dataset to train the classifier and pickle the trained model for future use. The model will be evaluated against the eval.csv dataset.

## Prediction

predict.py can be used to classify data by unpickling the trained model in the script.

## Flask Application

app.py can be run to create a JSON Rest API for classifying data.