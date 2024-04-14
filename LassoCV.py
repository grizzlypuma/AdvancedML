import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, LassoCV

os.chdir("C:/Users/kazja/OneDrive/Documents/Advanced ML Coursera/Regularization")
df = pd.read_csv("Energy_Efficiency_Overfit_Dataset_Updated.csv")

