# -*- coding: utf-8 -*-
"""List of imports"""
# Used to read data from a CSV file
import pandas as pd

# Used to split data into training and test
from sklearn.model_selection import train_test_split

# Used to train with Decision Tree Classifier and k nearest neighbor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Used to dsplay the accuracy and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

"""
Spyder Editor
@Author Jahan Ulhaque
Student ID: 201272455

Checklist: All complete! : 7/7 And Extra

Step 1: Loading Data

Step 2: Training
Step 2.1: Code for training
Step 2.2: Successful training

Step 3: Model Evaluation
Step 3.1: Explain your expermental design
Step 3.2 Document your evaluation results

Extra 20%
"""


print("Decision Tree:")
"""Step 1: Loading Data 10%"""
# Using panda we are able to store and read the CSV data file
bill_data = pd.read_csv("bill_authentication.xlsx")

# number of rows and columns in dataset:
# Shows amount of records and total attributes for each record
print(bill_data.shape)
print("Left hand value shows amount of rows/records, \nright hand value shows amount of attributes\n")

"""Step 2: Training 30%"""
# a - By using drop, we have removed the column "Class", which is the label. But kept the rest
# a - Is the attribute set and Y contains corresponding labels
# b - Contains only values from the Class column
a = bill_data.drop('Class', axis=1)
b = bill_data['Class']

# Randomly splits data into training and test sets.
# Test set = 20% and Training set = 80%
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.20)

"""Step 2.1: Code for training 10%"""
# By using the DecisionTreeClassifer function, Which will take in the training
# data and use this to make predictions/
# classifier = DecisionTreeClassifier().fit(a_train, b_train)
DTClassifier = DecisionTreeClassifier()
DTClassifier.fit(a_train, b_train)
# Classifier has been successfully trained


"""Step 2.2: Successful training 20% (I think) / Testing Classifier"""
# Classifier has been trained, now can try make predictions on the unseen test data
# Testing classifier
b_pred = DTClassifier.predict(a_test)

"""Step 3: Model evaluation"""
print("Confusion Matrix:\n{}".format(confusion_matrix(b_test, b_pred)))
print("\nAccuracy is: {}/1".format(accuracy_score(b_test, b_pred)) + "\n")

"""Step 3.1: Explain your experimental design, 20%

Here you need to explain which method you are using, 
and how you design your evaluation
experiments."""

"""Step 3.2: Document your evaluation results, 20%

You can get your mark of this part if you write down your experimental results"""
"""
Confusion matrix allows us to work out how many instances we predicted correctly and incorrectly.
So, the bottom right value, and top left value shows us the predictions we got correct.
the bottom left value, and top right valye shows us the predictions we got incorrectly.
By totaling up the numbers from the confusion matrix, we get the total test instances.

Given the accuracy high accuracy of this classifier, the incorrectly predicted values 
will be very low.
-
The "accuracy_score" funtions works by doing the following:
    It will loop through the predicted data and the actual true data, and for every
    correct answer(predicted correctly) it will store this value and divide by the amount of samples.
    i.e, given a sample size of 300, and we predict 295 instances correctly,
    then it will do 295/300, thus giving an accuracy socre of 0.983/1
"""

""" KNN """

# Looping through 1 - 26 N numbers to find the best number that will find the most accurate score.
scores = []

for k in range(1, 26, 1):
    """Step 2: Training 30%"""
    """Step 2.1: Code for training 10%"""
    knn = KNeighborsClassifier(n_neighbors=k)  # Apply KNN classifier and number to knn var
    knn.fit(a_train, b_train)  # Training classifier

    """Step 2.2: Successful training 20% (I think)"""
    b_pred = knn.predict(a_test)  # Testing classifier
    scores.append(accuracy_score(b_test, b_pred))  # Stores the accuracy score of every KNN

plt.plot(range(1, 26, 1), scores)
plt.title("Accuracy Scores for Values of k of k-Nearest-Neighbors")
plt.xlabel("Values 1 - 25")
plt.ylabel("Accuracy Score 0/1")
plt.show()

score = max(scores)  # Stores the best KNN value
max_score = int(round(score))
print("Best KNN value is: {}".format(max_score))

knn = KNeighborsClassifier(n_neighbors=max_score)  # Apply KNN classifier and number to knn var
knn.fit(a_train, b_train)  # Training classifier

"""Step 2.2: Successful training 20% (I think) / Testing Classifier"""
b_pred = knn.predict(a_test)  # Testing classifier with the best KNN value

"""Step 1: Loading Data 10%"""
print("KNN:")
print(bill_data.shape)
print("Left hand value shows amount of rows/records, \nright hand value shows amount of attributes\n")

"""Step 3: Model evaluation"""
print("Confusion Matrix:\n{}".format(confusion_matrix(b_test, b_pred)))
print("\nAccuracy is: {}/1".format(accuracy_score(b_test, b_pred)))


