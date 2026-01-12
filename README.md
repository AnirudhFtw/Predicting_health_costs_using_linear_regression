# Predicting Health Costs Using Linear Regression

## 📌 Project Overview
This project demonstrates how to use **Ridge Regression** (a type of regularized linear regression) to predict **medical expenses** from patient data.  
We:
1. Preprocess the dataset (one-hot encoding, polynomial feature expansion, scaling).  
2. Implement a custom Ridge Regression wrapper class.  
3. Train, predict, and evaluate the model.  

---

## 📂 Steps

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
