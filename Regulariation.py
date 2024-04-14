## PROBLEM STATEMENT
## Develop a predictive model using employee data to classify individuals as likely to stay or leave the company. This classification will assist in making informed decisions about employee retention strategies and workplace improvements.

## Overview
"""The dataset contains 900 rows and 15 columns, representing various employee metrics. The data aims to reflect realistic scenarios in a corporate setting, encompassing professional and personal employee metrics. Below is a brief overview of the dataset columns:

1. JobSatisfaction: Employee's job satisfaction level.

2. PerformanceRating: Performance rating given by the company.

3. YearsAtCompany: Total number of years the employee has been with the company.

4. WorkLifeBalance: Rating of how well the employee feels they balance work and personal life.

5. DistanceFromHome: Distance from the employee's home to the workplace.

6. MonthlyIncome: The monthly income of the employee.

7. EducationLevel: The highest level of education attained by the employee.

8. Age: The age of the employee.

9. NumCompaniesWorked: The number of companies the employee has worked at before joining the current company.

10. EmployeeRole: The role or position of the employee within the company.

11. AnnualBonus: Annual bonus received by the employee.

12. TrainingHours: Number of hours spent in training programs.

13. Department: Department in which the employee works.

14. AnnualBonus_Squared: Square of the annual bonus (a polynomial feature).

15. AnnualBonus_TrainingHours_Interaction: Interaction term between annual bonus and training hours. """

"""It is clear from the above description that the EmployeeTurnover is the 'Target' column.
Binary outcome variable, with '1' indicating the employee is likely 
to leave (turnover) and '0' indicating the employee 
is likely to stay. Understanding factors leading to 
turnover is crucial for the company to develop effective employee retention strategies and 
improve overall workplace satisfaction."""

"""Let us begin with importing the necessary libraries. And read the data."""

# Necessary library imports for data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, f1_score

# Load the dataset
df = pd.read_csv('modified_employee_turnover.csv')


# Dropping 'Target' column to avoid muticollinearity
# Write your code below
X = df.drop('Employee_Turnover', axis = 1)
y = df['Employee_Turnover']

# Splitting Dataset into Train and Test Sets


"""This step is a standard procedure in machine learning for preparing data before training a model. 
It ensures that there is a separate dataset for evaluating the model's performance, 
which helps in assessing how well the model will perform on unseen data. 
Split the data in 70:30 ratio. Name the variables as follows- X_train, X_test, y_train, y_test.
And use the random_state as 42."""

# Splitting the dataset into training and testing sets
X_train,X_test,y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# Logistic regression without regularization

logistic_no_reg = LogisticRegression(penalty= 'none',
                                    max_iter = 10000,
                                    n_jobs = -1)
logistic_no_reg.fit(X_train, y_train)

# Evaluate the model without regularization for train data
y_train_pred_no_reg = logistic_no_reg.predict(X_train)
f1_no_reg_train = f1_score(y_train, y_train_pred_no_reg)
print("F1 Score without Regularization:", f1_no_reg_train)
### OUTPUT:F1 Score without Regularization: 0.8667366211962225

# Evaluate the model without regularization for test data
y_test_pred_no_reg = logistic_no_reg.predict(X_test)
f1_no_reg_test = f1_score(y_test, y_test_pred_no_reg)
print("F1 Score without Regularization:", f1_no_reg_test)
### OUTPUT:F1 Score without Regularization: 0.8541666666666667

# Apply Logistic Regression with L1 Regularization

"""It's time to fit our model in 'logistic_l1_cv'."""
# Definig a range for Cs
Cs = np.linspace(0.001,10,20)


# Logistic regression with L1 regularization using cross-validation to find the best C

logistic_l1_cv = LogisticRegressionCV(Cs=Cs,
                                      penalty='l1',
                                      solver='liblinear',
                                      )

logistic_l1_cv.fit(X_train, y_train)

### Lets now examine the Regularization Strengths

logistic_l1_cv.Cs_
### OUTPUT:
"""array([1.00000000e-03, 5.27263158e-01, 1.05352632e+00, 1.57978947e+00,
       2.10605263e+00, 2.63231579e+00, 3.15857895e+00, 3.68484211e+00,
       4.21110526e+00, 4.73736842e+00, 5.26363158e+00, 5.78989474e+00,
       6.31615789e+00, 6.84242105e+00, 7.36868421e+00, 7.89494737e+00,
       8.42121053e+00, 8.94747368e+00, 9.47373684e+00, 1.00000000e+01])"""

best_C = logistic_l1_cv.C_
print(f"The best Cs value is: {best_C}")
### OUTPUT: The best Cs value is: [1.57978947]

## We can see that the peak comes at 1.57
"""Next apply Logistic regression with L1 regularization in 'logistic_l1_cv' using 
cross-validation(cv=5), and peak value 0.35 for C value.
Penalty would remain L1, solver would be liblinear and max_iter=10000"""

# Logistic regression with L1 regularization using cross-validation
logistic_l1_cv = LogisticRegressionCV(Cs = Cs,
                                      penalty='l1',
                                      solver='liblinear',
                                      cv = 5,
                                      max_iter = 10000)
logistic_l1_cv.fit(X_train, y_train)

# Evaluate the model with L1 regularization on train data
y_train_pred_l1 = logistic_l1_cv.predict(X_train)
f1_l1_train = f1_score(y_train, y_train_pred_l1)
print("F1 Score with L1 Regularization on train data :", f1_l1_train)
### OUTPUT:F1 Score with L1 Regularization on train data : 0.8718487394957983

# Evaluate the model with L1 regularization on train data
y_test_pred_l1 = logistic_l1_cv.predict(X_test)
f1_l1_test = f1_score(y_test, y_test_pred_l1)
print("F1 Score with L1 Regularization on test data :", f1_l1_test)
### OUTPUT:F1 Score with L1 Regularization on test data : 0.8511749347258485


# Apply Logistic Regression with L2 Regularization on train data
"""Next apply Logistic regression with L2 regularization in 'logistic_l2_cv' using cross-validation(cv=5), default C value."""

# Definig a range for Cs
Cs = np.linspace(0.001,10,20)


# Logistic regression with L2 regularization using cross-validation to find the best C

logistic_l2_cv = LogisticRegressionCV(Cs=Cs,
                                      penalty='l2',
                                      solver= 'liblinear',
                                      cv=5,
                                      max_iter=10000,
                                      n_jobs=-1)
logistic_l2_cv.fit(X_train, y_train)
logistic_l2_cv.Cs_

"""
array([1.00000000e-03, 5.27263158e-01, 1.05352632e+00, 1.57978947e+00,
       2.10605263e+00, 2.63231579e+00, 3.15857895e+00, 3.68484211e+00,
       4.21110526e+00, 4.73736842e+00, 5.26363158e+00, 5.78989474e+00,
       6.31615789e+00, 6.84242105e+00, 7.36868421e+00, 7.89494737e+00,
       8.42121053e+00, 8.94747368e+00, 9.47373684e+00, 1.00000000e+01])"""

logistic_l2_cv.C_
### OUTPUT: array([3.15857895])

## Peak at 3.15
"""Apply Logistic regression with L2 regularization in 'logistic_l2_cv' using cross-validation with peak value from the plot. """

# Logistic regression with L2 regularization using cross-validation to find the best C
#Write your code below
logistic_l2_cv = LogisticRegressionCV(Cs=Cs,
                                      penalty='l2',
                                      solver='liblinear',
                                      cv=5,
                                      max_iter=10000,
                                      n_jobs=-1)
logistic_l2_cv.fit(X_train, y_train)

# Evaluate the model with L2 regularization on old train data
y_train_pred_l2 = logistic_l2_cv.predict(X_train)
f1_l2_train = f1_score(y_train, y_train_pred_l2)
print("F1 Score with L2 Regularization:", f1_l2_train)
### OUTPUT:F1 Score with L2 Regularization: 0.865484880083420a3

# Evaluate the model with L2 regularization on old test data
y_test_pred_l2 = logistic_l2_cv.predict(X_test)
f1_l2_test = f1_score(y_test, y_test_pred_l2)
print("F1 Score with L2 Regularization:", f1_l2_test)
### OUTPUT:F1 Score with L2 Regularization: 0.8475452196382429

'''An F1 score of around 0.86 for one dataset and 0.847 for the other dataset in the context of 
L2 regularization indicates that your logistic regression model performs well on both the dataset 
it was trained on and on new, unseen data. The scores suggest that the model is accurately predicting 
the target variable, maintaining a balance between precision and recall, and the L2 regularization is 
likely helping to enhance the model's ability to generalize.'''

## ElasticNet Regularization
'''Apply Logistic regression with ElasticNet regularization in 'logistic_en_cv' using cross-validation'''

# Definig a range for Cs
Cs = np.linspace(0.001,10,20)

# Logistic regression with ElasticNet regularization using cross-validation
logistic_en_cv = LogisticRegressionCV(penalty='elasticnet',
                                      Cs = Cs,
                                      l1_ratios= [0.0001, 0.001, 0.01, 0.05, 0.1, 0.4, 0.5, 0.7, 1],
                                      solver='saga',
                                      cv=5,
                                      max_iter=1000000,
                                      n_jobs=-1)
logistic_en_cv.fit(X_train, y_train)

# Evaluate the model
y_train_pred_elastic = logistic_en_cv.predict(X_train)
f1_elastic_train = f1_score(y_train, y_train_pred_elastic)
print("F1 Score with Elastic Net Regularization on Train Set:", f1_elastic_train)
### OUTPUT: F1 Score with Elastic Net Regularization on Train Set: 0.8743400211193242

y_test_pred_elastic = logistic_en_cv.predict(X_test)
f1_elastic_test = f1_score(y_test, y_test_pred_elastic)
print("F1 Score with Elastic Net Regularization on Test Set:", f1_elastic_test)
### OUTPUT: F1 Score with Elastic Net Regularization on Test Set: 0.8443271767810026


### END
