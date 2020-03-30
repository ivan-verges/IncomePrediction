def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#Loads income data from dataset into a dataframe
income_data = pd.read_csv("income.csv", header = 0, delimiter = ", ")

#Creates a set of Countries from dataframe
countries = {}
i = 0
for value in income_data["native-country"]:
  if value not in countries:
    countries[value] = i
    i += 1

#Creates a new column into Dataframe to hold numeric values for each value in the Sex column
income_data["sex-int"] = income_data["sex"].apply(lambda row : 0 if row == "Male" else 1)

#Creates a new column into Dataframe to hold numeric values for countries in th Native-Country column
income_data["country-int"] = income_data["native-country"].apply(lambda row : 0 if row == "United-States" else 1)

#Creates a dataset with only the desired features
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]]

#Creates a dataset with only the desired labels to predict
labels = income_data[["income"]]

#Splits the data into Train Data, Test Data, Train Labels and Test Labels
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

#Creates a Random Forest model to train
forest = RandomForestClassifier(random_state = 1)

#Trains the model with data and labels
forest.fit(train_data, train_labels)

#Prints the Importance Value for the Features
print(forest.feature_importances_)

#Prints the Score of the model on the Test data and labels
print(forest.score(test_data, test_labels))

#Prints the Predictions of the model for the Test data
print(forest.predict(test_data))