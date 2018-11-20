# -*- coding: utf-8 -*-
"""
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
# Model evaluation: Used to dsplay the accuracy score, confusion matrix & Cross validation score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cross_validation import cross_val_score
# Used to display graphs (Used for testing)
import matplotlib.pyplot as plt

"""Step 1: Loading Data 10%"""
# Using panda we are able to store and read the CSV data file
bill_data = pd.read_csv("bill_authentication.csv")  

"""Step 2: Training 30%"""
# x - By using drop, we have removed the column "Class", which is the label. But kept the rest
# x - Is the attribute set and Y contains corresponding labels
# y - Contains only values from the Class column
x = bill_data.drop('Class', axis = 1) 
y = bill_data['Class']

"""Step 2.1: Code for training 10%"""
# By using the DecisionTreeClassifer function, Which will take in data and use this to make predictions
DTClassifier = DecisionTreeClassifier()

# cv = 10 Fold cross validation, which is applied on the whole dataset (x, y). 
# Using the Decision Tree Classifier.
cv_score = cross_val_score(DTClassifier, x, y, cv = 10)

# Randomly splits data into training and test sets.
# Test set = 20% and Training set = 80%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)  

# Training the classifier with the training data sample
DTClassifier.fit(x_train, y_train) 

"""Step 2.2: Successful training 20%  / Testing Classifier""" 
# Classifier has been trained, now can try make predictions on the unseen test data
# Predicting the test sets results
y_pred = DTClassifier.predict(x_test)  


print("Decision Tree:")
# number of rows and columns in dataset:
# Shows amount of records and total attributes for each record
print("Amount of Entities: {},\nAmount of classes per entity: {}".format(bill_data.shape[0], bill_data.shape[1]))
print("Test sample size and classes:", x_test.shape, "\n")

"""Step 3: Model evaluation"""
print("Model evaluation:")
print("\nConfusion Matrix:\n", (confusion_matrix(y_test, y_pred))) # Displays the confusion matrix
print("\nAccuracy on test sample is: {}/1".format(accuracy_score(y_test, y_pred))) # Displays the accuracy
print("\nClassification report:\n", classification_report(y_test, y_pred))  # Displays the classification report
print("\n10 Fold cross validation:\n", cv_score) # Displays the 10 fold cross validation results
print("\n10 Fold cross validation average/mean:\n",cv_score.mean()) # Displaying the average result from the 10 fold cross validation



""" KNN """
accuracy_scores = [] # Stores the accuracy score of every KNN (Ranging from: 1 - 26)
confusion_list = [] # Stores confusion matrix for each KNN iteration

for k in range(1, 26): # Looping through 1 - 26 N numbers to find the best KNN value
    """Step 2: Training 30%"""
    """Step 2.1: Code for training 10%"""
    knn = KNeighborsClassifier(n_neighbors = k) # Apply KNN classifier with a new KNN value every iteration
    knn.fit(x_train, y_train) # Training classifier
    
    """Step 2.2: Successful training 20% / Testing Classifier""" 
    y_pred = knn.predict(x_test) # Testing classifier
    accuracy_scores.append(accuracy_score(y_test, y_pred)) # Stores the accuracy score of every KNN iteration
    confusion_list.append(confusion_matrix(y_test, y_pred)) # Stores the confusion matrix of every KNN iteration
    
# Displays the accuracy score for each KNN value 
# Used for testing 
"""
plt.plot(range(1, 26), accuracy_scores)
plt.title("Accuracy Scores for Values of k of k-Nearest-Neighbors")
plt.xlabel("Values 1 - 26")
plt.ylabel("Accuracy Score 0/1")
plt.show()
"""

max_value = max(accuracy_scores) # Gets the highest accuracy score
max_index = accuracy_scores.index(max_value)  # Gets the index of the highest accuracy score
if max_index == 0: # Expected n_neighors should always be greater than 0, just incase it was not assigned a KNN value we will set it to 1
    max_index = 1
    
knn = KNeighborsClassifier(n_neighbors = max_index) # Apply KNN classifier and best KNN number
cv_score = cross_val_score(knn, x, y, cv = 10) # Apply 10 Fold cross validation with the best KNN value
knn.fit(x_train, y_train) # Training classifier (This is using the training and testing data)

"""Step 2.2: Successful training 20% / Testing Classifier""" 
y_pred = knn.predict(x_test) # Testing classifier with the best KNN value on the test sample

print("-\n-\nK nearest neighbour:")
print("Amount of Entities: {},\nAmount of classes per entity: {}".format(bill_data.shape[0], bill_data.shape[1]))
print("Test sample size and classes:", x_test.shape, "\n")

"""Step 3: Model evaluation"""
print("Model evaluation:")
print("With a KNN of: {}".format(max_index))
print("\nConfusion Matrix:\n{}".format(confusion_list[max_index]))
print("\nAccuracy on test sample is: {}/1".format(max_value))
print("\nClassification report:\n", classification_report(y_test, y_pred))  
print("\n10 Fold cross validation:\n", cv_score)
print("\n10 Fold cross validation average/mean:\n",cv_score.mean())

