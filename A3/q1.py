import csv
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import os

# reading the data
data = []
with open('spam_ham_dataset.csv', newline='') as file:
    csv_file_reader = csv.reader(file)
    for row in csv_file_reader:
        data.append(row)


# extracting relevant columns from the data
data = data[1:]
data = [inner_list[2:] for inner_list in data]

nonspam = 0
spam = 0
for i in data:
    if i[1] == '0':
        nonspam += 1
    else:
        spam += 1

for i in data:
    if i[1] == '0':
        i[1] = 0
    else:
        i[1] = 1

print(spam)
print(nonspam)
print(nonspam/len(data)*100)
print(spam/len(data)*100)

# separating the data into train and test sets, with the same proportion of spam and not spam as the original data

features = [i[0] for i in data]  # Extract features (data)
labels = [i[1] for i in data]    # Extract labels
train_ratio = 0.8
num_samples = len(data)
# Assuming 'labels' is a list of class labels corresponding to each data point
train_data, test_data, train_labels, test_labels = train_test_split(features, labels, train_size=train_ratio, stratify=labels, random_state=42)

train_spam = 0
train_nonspam = 0
test_spam = 0
test_nonspam = 0
for i in train_labels:
    if i == 1:
        train_spam += 1
    else:
        train_nonspam += 1
for i in test_labels:
    if i == 1:
        test_spam += 1
    else:
        test_nonspam += 1
print(train_spam)
print(train_nonspam)
print(test_spam)
print(test_nonspam)

# function to preprocess the data
def preprocess(text):
    # cleaning the text. removing tags, punctuation, and special characters, so only words exist in the emails. 
    pattern = r'[\n\t\r\b\f\\]'
    digit = r'\b\d+\b'
    single_chars = r'\b[a-zA-Z]\b'
    # removing \characters
    text = re.sub(pattern, ' ', text)
    # removing punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # removing stopwords
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and len(word) > 3]
    text = ' '.join(filtered_words)
    # removing "subject"
    if text.startswith("subject"):
        text = text[len("subject"):].lstrip()
    # removing digits
    text = re.sub(digit, '', text)
    # removing single chars
    text = re.sub(single_chars, '', text)
    return text


# preprocessing the train data
preprocessed_data = []
for text in train_data:
    preprocessed_data.append(preprocess(text))    

# preprocessing the test data
test_preprocessed_data = []
for text in test_data:
    test_preprocessed_data.append(preprocess(text))

# implementing svm
# the data features are represented using tfidf vectorizer
combined_data = preprocessed_data + test_preprocessed_data
tfidf = TfidfVectorizer(vocabulary=None)
tfidf.fit(combined_data)
x_train = tfidf.transform(preprocessed_data)
x_test = tfidf.transform(test_preprocessed_data)
model = SVC(kernel='rbf')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, x_train, train_labels, cv=kfold, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
model.fit(x_train, train_labels)
print(model.score(x_test,test_labels))

# testing with different values of hyperparameters and kernels in svm
models = [
    SVC(kernel='rbf'),
    SVC(kernel='linear'),
    SVC(kernel='poly', degree=2),
    SVC(kernel='poly', degree=3),
    SVC(C=2, kernel='rbf'),
    SVC(C=10, kernel='rbf'),
    SVC(C=25, kernel='linear')
]

model_names = [
    'SVM (RBF)',
    'SVM (Linear)',
    'SVM (Poly, degree=2)',
    'SVM (Poly, degree=3)',
    'SVM (C=2, RBF)',
    'SVM (C=10, RBF)',
    'SVM(C=25, Linear)'
]

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mean_accuracies = []
for model in models:
    cv_scores = cross_val_score(model, x_train, train_labels, cv=kfold, scoring='accuracy')
    mean_accuracy = cv_scores.mean()
    mean_accuracies.append(mean_accuracy)

plt.figure(figsize=(10, 6))
plt.barh(model_names, mean_accuracies, color='maroon')
plt.xlabel('Mean Accuracy')
plt.title('Mean Cross-Validation Accuracy for Different SVM Models')
plt.xlim(0.5, 1.0)  
plt.gca().invert_yaxis() 
plt.show()

# implementing logistic regression

# different kind of extracting in the case of logistic regression
all_words_list = []
for sentence in combined_data:
    sentence_tokens = sentence.split()
    for token in sentence_tokens:
        if token not in all_words_list:
            all_words_list.append(token)

all_words_set = {word: idx for idx, word in enumerate(all_words_list)}

# getting the counts of each word as features

def feature_extraction(data, all_words_set):
    points = []
    for i in data:
        point_vec = [0] * (len(all_words_set))
        i_tokens = i.split()
        for token in i_tokens:
            if token in all_words_set:
                index = all_words_set[token]
                point_vec[index] += 1
        points.append(point_vec)
    return points

log_points = feature_extraction(train_data, all_words_set)


# logistic regression 
def sigmoid(z):
    f = 1 / (1 + np.exp(-z))
    return f

def logistic_regression(data, labels, step_size, iterations):
    w = [0] * len(data[0])
    for i in range(iterations):
        grad_w = 0
        for j in range(0, len(data)):
            z = np.dot(data[i], w)
            error = labels[i] - sigmoid(z)
            grad_w += data[i] * error
        w += step_size * grad_w
    return w


log_points_np = np.array(log_points)
w_log = logistic_regression(log_points_np, train_labels, 0.01, 100)

# testing the model against the test set
def logistic_regression_model(data, w_log):
    all_points_features = feature_extraction(data, all_words_set)
    predicted_labels = []
    for point in all_points_features:
        z = np.dot(point, w_log)
        if (sigmoid(z) > 0.5):
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    return predicted_labels

result = logistic_regression_model(test_data, w_log)
accuracy = accuracy_score(test_labels, result)
print(accuracy)

# since the above functions take the whole dataset at once, the below functions give the prediction for a single test point
def single_feature_extraction(point, all_words_set):
    point_vec = [0] * (len(all_words_set))
    point_tokens = point.split()
    for token in point_tokens:
        if token in all_words_set:
            index = all_words_set[token]
            point_vec[index] += 1
    return point_vec

def single_logistic_regression_model(data_point, w_log):
    point_features = single_feature_extraction(data_point, all_words_set)
    z = np.dot(point_features, w_log)
    if (sigmoid(z) > 0.5):
        return 1
    else:
        return 0

# reading from the test folder as mentioned in the assignment

def test_folder_access():
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    sibling_dir = os.path.join(current_dir, 'test')
    data_files = os.listdir(sibling_dir)
    for filename in data_files:
        file_path = os.path.join(sibling_dir, filename)
        # Perform operations on the file (e.g., read content, process data, etc.)
        with open(file_path, 'r') as file:
            content = file.read()
            test_point = preprocess(content)
            svm_point = tfidf.transform([test_point])
            model = SVC(kernel='rbf')
            model.fit(x_train, train_labels)
            print(f"File '{filename}' SVM prediction: '{model.predict(svm_point)}'")
            log_pred = single_logistic_regression_model(test_point, w_log)
            print(f"File '{filename}' Logistic Regression prediction: '{log_pred}'")

test_folder_access()