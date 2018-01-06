from sklearn.externals import joblib
text_clf_svm = joblib.load('C:/Repo/python_tests/scikit_tests/hackernews/text_clf_svm_model.pkl') 

test_input = ["Microsoft Azure launches new development environment"]
predicted_svm = text_clf_svm.predict(test_input)
print(predicted_svm)

