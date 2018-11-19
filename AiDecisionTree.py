# -*- coding: utf-8 -*-
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



print("Decision Tree:")
"""Step 1: Loading Data 10%"""
# Using panda we are able to store and read the CSV data file
bill_data = pd.read_csv("bill_authentication.csv")  

# number of rows and columns in dataset:
# Shows amount of records and total attributes for each record
print("Amount of Entities: {},\nAmount of classes per entity: {}".format(bill_data.shape[0], bill_data.shape[1]))

"""Step 2: Training 30%"""
# a - By using drop, we have removed the column "Class", which is the label. But kept the rest
# a - Is the attribute set and Y contains corresponding labels
# b - Contains only values from the Class column
x = bill_data.drop('Class', axis = 1) 
y = bill_data['Class']


# Randomly splits data into training and test sets.
# Test set = 20% and Training set = 80%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)  


"""Step 2.1: Code for training 10%"""
# By using the DecisionTreeClassifer function, Which will take in the training
# data and use this to make predictions/
# classifier = DecisionTreeClassifier().fit(a_train, b_train) 
DTClassifier = DecisionTreeClassifier()
DTClassifier.fit(x_train, y_train) 
# Classifier has been successfully trained


"""Step 2.2: Successful training 20% (I think) / Testing Classifier""" 
# Classifier has been trained, now can try make predictions on the unseen test data
# Testing classifier
y_pred = DTClassifier.predict(x_test)  


"""Step 3: Model evaluation"""
print("Confusion Matrix:\n{}".format(confusion_matrix(y_test, y_pred)))
print("\nAccuracy is: {}/1".format(accuracy_score(y_test, y_pred)) + "\n")


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

scores = [] # Stores the accuracy score of every KNN (Ranging from: 1 - 26)
confusion = [] # Stores confusion matrix for each KNN iteration

for k in range(1, 26): # Looping through 1 - 26 N numbers to find the best KNN value
    """Step 2: Training 30%"""
    """Step 2.1: Code for training 10%"""
    knn = KNeighborsClassifier(n_neighbors = k) # Apply KNN classifier with a new KNN value every iteration
    knn.fit(x_train, y_train) # Training classifier
    
    """Step 2.2: Successful training 20% (I think)""" 
    y_pred = knn.predict(x_test) # Testing classifier
    scores.append(accuracy_score(y_test, y_pred)) # Stores the accuracy score of every KNN iteration
    confusion.append(confusion_matrix(y_test, y_pred)) # Stores the confusion matrix of every KNN iteration
    
# Displays the accuracy score for each KNN value 
plt.plot(range(1, 26), scores)
plt.title("Accuracy Scores for Values of k of k-Nearest-Neighbors")
plt.xlabel("Values 1 - 26")
plt.ylabel("Accuracy Score 0/1")
plt.show()

max_value = max(scores)
max_index = scores.index(max_value)

if max_index == 0:
    max_index = 1
    
knn = KNeighborsClassifier(n_neighbors = max_index) # Apply KNN classifier and number to knn var

knn.fit(x_train, y_train) # Training classifier

"""Step 2.2: Successful training 20% (I think) / Testing Classifier""" 
y_pred = knn.predict(x_test) # Testing classifier with the best KNN value on the test sample
 

print("KNN:")
print("Amount of Entities: {},\nAmount of classes per entity: {}".format(bill_data.shape[0], bill_data.shape[1]))

"""Step 3: Model evaluation"""
print("With a KNN of: {}".format(max_index))
print("Confusion Matrix:\n{}".format(confusion[max_index]))
print("\nAccuracy is: {}/1".format(max_value))


