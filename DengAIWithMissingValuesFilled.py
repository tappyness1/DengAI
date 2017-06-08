from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi
import scipy.stats
import matplotlib.pyplot as plt

"""modelname = smf.ols(formula='QUANT_RESPONSE ~ C(CAT_EXPLANATORY)', data = dataframe.fit())"""
#read csv and set variable 'data' to the read file
data = pd.read_csv('dengue_features_train_1.csv', low_memory = 'False', sep = ',')
data1 = pd.read_csv('dengue_features_test.csv', low_memory = 'False', sep = ',')
# change data index to numeric instead of strings 
# coerce basically forces the blanks to be NaN
data.apply(lambda x: pd.to_numeric(x, errors='ignore'))
data1.apply(lambda x: pd.to_numeric(x, errors='ignore'))
# if we handle missing data, cannot dropna
# data_clean = data.dropna()
# data1_clean = data1.dropna()

# the HandleMissingValues function fills NaN in column with the median of the 
# column

def HandleMissingValues(col):
    data[col] = data[col].fillna(data[col].median())
    # print (data[col])
def HandleMissingValues1(col):
    data1[col] = data1[col].fillna(data1[col].median())
# apply HandleMissingValues fn to all the columns
x = list(data)
for i in x[4:-2]:
    HandleMissingValues(i)


y = list(data1)
for i in y[4:-1]:
    HandleMissingValues1(i)    
# print (data.head(n=5))
# print (data1.head(n=20))

# centre the variables to make them comparable
# def centrevar(x):
#     # y = x + '_c'
#     data_clean[x] = (data_clean[x] - data_clean[x].mean())
#    
# x = list(data_clean)
# y = []
# for i in x[4:-2]:
#     centrevar(i)
#     # to_list = i + '_c'
#     # y.append(to_list)
# z = list(data1_clean)
# for i in z[4:]:
#     centrevar(i)    
# regstr = 'total_cases ~ ' + ' + '.join(y[:-1])
# 
# reg2 = smf.ols(formula = regstr, data = data_clean).fit()
# print (list(data_clean))
# 
# print (reg2.summary())

# datasub = data_clean[data_clean.columns[4:-2]]
# datasub1 = data_clean[['ndvi_nw', 'reanalysis_avg_temp_k','reanalysis_specific_humidity_g_per_kg']]
# data1sub = data1_clean[data1_clean.columns[4:-1]]
# data1sub1 = data1_clean[['ndvi_nw', 'reanalysis_avg_temp_k','reanalysis_specific_humidity_g_per_kg']]
# 
# # print (list(datasub))

# ML part

datasub = data[data.columns[4:-2]]
data1sub = data1[data1.columns[4:-1]]

trainingData = np.array(datasub)
trainingScores = np.array(data['total_cases'])
lassoreg = Lasso(alpha=1.0)
lassoreg.fit(trainingData,trainingScores)

predictionData = np.array(data1sub)
y_pred = lassoreg.predict(predictionData)

result = []
for i in y_pred:
    result.append(int(i))
print (len(result))
data1['total_cases'] = result
y_pred = data1['total_cases']
print (y_pred.dtypes)

#concat the two dfs to make it one df for csv file
predicted_df = data1[['city', 'year', 'weekofyear','total_cases']]
df1 = data1[['city', 'year', 'weekofyear']]
# print (df1.dtypes)
final_pred = pd.concat([df1,y_pred], axis = 1)
print (final_pred.total_cases)
# final_pred['total_cases'] = final_pred['total_cases'].fillna(0)
final_pred['total_cases'] = final_pred['total_cases'].astype(int)
print(final_pred.dtypes)
print (final_pred.head(n= 20))

# write to csv
final_pred.to_csv('test.csv', sep = ',', index = False)

