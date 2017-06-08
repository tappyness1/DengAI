from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV
from sklearn import preprocessing

#read csv and set variable 'data' to the read file
data = pd.read_csv('dengue_features_train_1.csv', low_memory = 'False', sep = ',')
data1 = pd.read_csv('dengue_features_test.csv', low_memory = 'False', sep = ',')
# change data index to numeric instead of strings 
# coerce basically forces the blanks to be NaN
data.apply(lambda x: pd.to_numeric(x, errors='ignore'))
data1.apply(lambda x: pd.to_numeric(x, errors='ignore'))
# if we handle missing data, cannot dropna
data = data.dropna()
# data1_clean = data1.dropna()

# the HandleMissingValues function fills NaN in column with the median of the 
# column
 
# def HandleMissingValues(col):
#     data[col] = data[col].fillna(data[col].median())
#     # print (data[col])
# def HandleMissingValues1(col):
#     data1[col] = data1[col].fillna(data1[col].median())
# # apply HandleMissingValues fn to all the columns
# x = list(data)
# for i in x[4:-2]:
#     HandleMissingValues(i)

# y = list(data1)
# for i in y[4:-1]:
#     HandleMissingValues1(i)    

# ML part

# set up predictors and target
predvar = data[data.columns[4:-2]]

targets = data.total_cases

predictors = predvar.copy()

for var in predictors:
    predictors[var] = preprocessing.scale(predictors[var].astype('float64'))

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, targets, test_size = 0.4, random_state = 123)

model = LassoLarsCV(cv = 10, precompute=False).fit(pred_train, tar_train)

printout = dict(zip(predictors.columns, model.coef_))
import json

print (json.dumps(printout, indent =4 ))

# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')
plt.show()

# plot MSE
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')

plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.legend()
plt.title('Mean squared error on each fold')
plt.show()

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE: ' + str(train_error))
print ('test data MSE: ' + str(test_error))

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square: '+ str(rsquared_train))
print ('test data R-square: '+ str(rsquared_test))