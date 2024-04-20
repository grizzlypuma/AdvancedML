
# # PROBLEM STATEMENT
# Anova Insurance, a global health insurance company, seeks to optimize its insurance policy premium pricing based on the health status of applicants. Understanding an applicant's health condition is crucial for two key decisions:
# - Determining eligibility for health insurance coverage.
# - Deciding on premium rates, particularly if the applicant's health indicates higher risks.
#
# Your objective is to Develop a predictive model that utilizes health data to classify individuals as 'healthy' or 'unhealthy'. This classification will assist in making informed decisions about insurance policy premium pricing.

# # OVERVIEW
#
# The dataset contains 10,000 rows and 20 columns (original data without preprocessing), the no. of columns becomes 23 post preprocessing because of encoding, the 23 columns includes both numerical and categorical variables. Here is the data dictionary.
#
# - Age: Represents the age of the individual. Negative values seem to be present, which might indicate data entry errors or a specific encoding used for certain age groups.
#
# - BMI (Body Mass Index): A measure of body fat based on height and weight. Typically, a BMI between 18.5 and 24.9 is considered normal.
#
# - Blood_Pressure: Represents systolic blood pressure. Normal blood pressure is usually around 120/80 mmHg.
#
# - Cholesterol: This is the cholesterol level in mg/dL. Desirable levels are usually below 200 mg/dL.
#
# - Glucose_Level: Indicates blood glucose levels. It might be fasting glucose levels, with normal levels usually ranging from 70 to 99 mg/dL.
#
# - Heart_Rate: The number of heartbeats per minute. Normal resting heart rate for adults ranges from 60 to 100 beats per minute.
#
# - Sleep_Hours: The average number of hours the individual sleeps per day.
#
# - Exercise_Hours: The average number of hours the individual exercises per day.
#
# - Water_Intake: The average daily water intake in liters.
#
# - Stress_Level: A numerical representation of stress level.
#
# - Target: This is a binary outcome variable, with '1' indicating 'Unhealthy' and '0' indicating 'Healthy'.
#
# - Smoking: A categorical variable indicating smoking status. Contains values - (0,1,2) which specify the regularity of smoking with 0 being no smoking and 2 being regular smmoking.
#
# - Alcohol: A categorical variable indicating alcohol consumption status. Contains values - (0,1,2) which specify the regularity of alcohol consumption with 0 being no consumption quality and 2 being regular consumption.
#
# - Diet: A categorical variable indcating the quality of dietary habits. Contains values - (0,1,2) which specify the quality of the habit with 0 being poor diet quality and 2 being good quality.
#
# - MentalHealth: Possibly a measure of mental health status. Contains values - (0,1,2) which specify the severity of the mental health with 0 being fine and 2 being highly severe
#
# - PhysicalActivity: A categorical variable indicating levels of physical activity. Contains values - (0,1,2) which specify the instensity of the medical history with 0 being no Physical Activity and 2 being regularly active.
#
# - MedicalHistory: Indicates the presence of medical conditions or history. Contains values - (0,1,2) which specify the severity of the medical history with 0 being nothing and 2 being highly severe.
#
# - Allergies: A categorical variable indicating allergy status. Contains values - (0,1,2) which specify the severity of the allergies with 0 being nothing and 2 being highly severe.
#
# - Diet_Type: Categorical variable indicating the type of diet an individual follows. Contains values(Vegetarian, Non-Vegetarian, Vegan).
# - (this column has been encoded into three different columns during the preprocessing stage)
#  - Diet_Type_Vegan,Diet_Type_Vegetarian
#
# - Blood_Group: Indicates the blood group of the individual Contains values (A, B, AB, O), this column values are encoded too .
#
# It is clear from the above description that the predictor variable is the 'Target' column.
#

# # -----------------------------------------------------------------------------

# ## Guidelines to follow in this notebook
# - The name of the dataframe should be df
# - Keep the seed value 42
# - Names of training and testing variables should be X_train, X_test, y_train, y_test
# - Keep the name of model instance as "model", e.g. model = DecisionTreeClassifer()
# - Keep the predictions on training and testing data in a variable named y_train_pred and y_test_pred respectively.

# # -------------------------------------------------------------------------------

#
# Let us begin with importing the necessary libraries.



#import relevant data libraries
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
get_ipython().system('pip install xgboost')
import xgboost as xgb
get_ipython().system('pip install lightgbm')
from lightgbm import LGBMClassifier
get_ipython().system('pip install catboost')
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

# ## Load the dataset

#Load the dataset
df = pd.read_csv('Healthcare_Dataset_Preprocessednew.csv')
df.head()



#shape of data
df.shape

# OUTPUT: (9549, 23)

# Column names in the dataset
df.columns

#OUTPUT:
"""Index(['Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Glucose_Level', 'Heart_Rate', 'Sleep_Hours', 'Exercise_Hours', 'Water_Intake', 
'Stress_Level', 'Target', 'Smoking', 'Alcohol', 'Diet', 'MentalHealth', 'PhysicalActivity', 'MedicalHistory', 'Allergies', 
'Diet_Type__Vegan', 'Diet_Type__Vegetarian', 'Blood_Group_AB', 'Blood_Group_B', 'Blood_Group_O'], dtype='object')"""

# # Separate the indpendent features in the dataframe 'X' and  the target in a variable 'y '

X = df.drop('Target', axis = 1)
y = df['Target']

X.shape , y.shape

#OUTPUT: ((9549, 22), (9549,))

# # Splitting Dataset into Train and Test Sets

# Splitting the dataset into training and testing sets keeping the test size as 25% and seed value as 42

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

# # Create a simple Decision tree
#
# The first step is to train the model using simple decision tree

# Initialize a simple Decision Tree classifier with depth 15 and seed 42. Name it 'model' and then fit it

model = DecisionTreeClassifier(max_depth=15, random_state=42)


## Begin hidden test
assert model.max_depth == 15, "Max_depth is not set to 15"
## End hidden test


model.fit(X_train, y_train)
# # Evaluate model performance

# After creating model and getting the predictions on the test set,
#  calculate f1 score for evaluating the performance of the model



y_pred_train = model.predict(X_train)

f1 = f1_score(y_train, y_pred_train)
print(f"The F1 score for Decision Tree is {f1}")

#OUTPUT: The F1 score for Decision Tree is 0.9907983761840325

# # APPLY ADABOOST ALGORITHM FOR TRAINING

# After creating simple decision tree, its now time to create classifier model using AdaBoostClassifier

model = AdaBoostClassifier()

# Train the AdaBoost model
model.fit(X_train, y_train)

# After training the model using AdaBoostClassifier , do prediction on the test data and calculate f1 score for evaluation

# Predict on training and validation data
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)



f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)

print(f"F1 score for train set is {f1_train}")
print(f"F1 score for test set is {f1_test}")

#OUTPUT: F1 score for train set is 0.8472296933835396
#OUTPUT: F1 score for test set is 0.8648860958366065

# # APPLY GRADIENT BOOSTING
#
# ### Now use GradientBoostingClassifier for training the model
#

model = GradientBoostingClassifier()
model.fit(X_train, y_train)


# ## Evaluation

# Prediciton on training and testing data using GradientBoostingClassifier and then calculate f1 score

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)

print(f"F1 score for train set is {f1_train}")
print(f"F1 score for test set is {f1_test}")

#OUTPUT: F1 score for train set is 0.9318728292813251
#OUTPUT: F1 score for test set is 0.9256198347107438

# # APPLY XGBOOST
#
# ### Here, use XGBClassifier to train the model.
#

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)

print(f"F1 score for train set is {f1_train}")
print(f"F1 score for test set is {f1_test}")

#OUTPUT: F1 score for train set is 0.9985165205664194
#OUTPUT: F1 score for test set is 0.9503154574132492




# # APPLY LIGHTGBM
#
# ### TRAIN MODEL WITH LGBMClassifier this time
model = LGBMClassifier()
model.fit(X_train, y_train)

# Predict on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

from sklearn.metrics import f1_score
f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)

print(f"F1 score for train set is {f1_train}")
print(f"F1 score for test set is {f1_test}")

#OUTPUT: F1 score for train set is 0.9985165205664194
#OUTPUT: F1 score for test set is 0.9503154574132492



# # APPLY CATBOOST
#
# ### Train model with CatBoostClassifier

model = CatBoostClassifier()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

f1_train = f1_score(y_train, y_pred_train)
f1_test = f1_score(y_test, y_pred_test)

print(f"F1 score for train set is {f1_train}")
print(f"F1 score for test set is {f1_test}")

#OUTPUT: F1 score for train set is 0.9985165205664194
#OUTPUT: F1 score for test set is 0.9503154574132492



# The performance of various boosting techniques depends on multiple hyperparamters and dataset provided for training.
# With gradient boosting, good precision can be achieved in reducing errors. There are multiple advantages in using gradient boosting as compared to adaboost. Also XgBoost is again evolved version of gradient boosting. Being versatile and capability of handling large datasets makes it more popular. While CatBoost is popular to handle categorical variables more vigourously and doesnt require encoding for the same.
#