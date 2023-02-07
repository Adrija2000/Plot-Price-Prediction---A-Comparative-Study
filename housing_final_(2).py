#Import libaries 

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import scipy.stats
import pylab
import statsmodels.api as sm
from statsmodels.regression import linear_model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing

"""## DATASET"""

data = pd.read_csv("Housing Data 3.csv")
data
data1 = data.drop("Localities" , axis = 1)
data1

data1.info()

data1.head()

data1.describe()

x1 = data1["No. of metro stations"]
x2 = data1["No. of railway stations(<12kms)"]
x3 = data1["No of bus stoppages"]
x4 = data1["Distance from the Airport(kms)"]
x5 = data1["No of health care centers(Hospitals/Pharmacy)"]
x6 = data1["No of Educational Institutes"]
x7 = data1["No of Departmental Stores"]
x8 = data1["No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)"]
x9 = data1["No of Restaurants"]
x10 = data1["No of Office"]
x11 = data1["No Religious Places(Temples/Mosques)"]
x12 = data1["No of Banks/ATMS"]
y = data1["Avg Price per sqft (in Rs.)"]

x = data1.drop("Avg Price per sqft (in Rs.)" , axis = 1)
x = x.astype(int)
x

"""#### DENSITY PLOT OF X"""

fig, axs = plt.subplots(2,6, figsize=(20,6))

sns.distplot(x1, ax=axs[0][0])
axs[0][0].set_title("Density plot of X1")

sns.distplot(x2, ax=axs[0][1])
axs[0][1].set_title("Density plot of X2")

sns.distplot(x3, ax=axs[0][2])
axs[0][2].set_title("Density plot of X3")

sns.distplot(x4, ax=axs[0][3])
axs[0][3].set_title("Density plot of X4")

sns.distplot(x5, ax=axs[0][4])
axs[0][4].set_title("Density plot of X5")

sns.distplot(x6, ax=axs[0][5])
axs[0][5].set_title("Density plot of X6")

sns.distplot(x7, ax=axs[1][0])
axs[1][0].set_title("Density plot of X7")

sns.distplot(x8, ax=axs[1][1])
axs[1][1].set_title("Density plot of X8")

sns.distplot(x9, ax=axs[1][2])
axs[1][2].set_title("Density plot of X9")

sns.distplot(x10, ax=axs[1][3])
axs[1][3].set_title("Density plot of X10")

sns.distplot(x11, ax=axs[1][4])
axs[1][4].set_title("Density plot of X11")

sns.distplot(x12, ax=axs[1][5])
axs[1][5].set_title("Density plot of X12")

plt.tight_layout()

"""#### NORMALITY CHECKING USING SHAPIRO WILK TEST"""

stat1 , pvalue1 = scipy.stats.shapiro(x1)
print(stat1)
print(pvalue1)

if pvalue1<=0.05:
    print("our data does not follow normal distribution")
else:
    print("our data follows normal distribution")

"""#### QQ PLOT"""

for i in x:
    print(x[i])
    scipy.stats.probplot(x[i] , dist = "norm" , plot = pylab)

scipy.stats.probplot(x1 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x2 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x3 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x4 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x5 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x6 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x7 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x8 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x9 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x10 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x11 , dist = "norm" , plot = pylab)
scipy.stats.probplot(x12 , dist = "norm" , plot = pylab)

plt.show()

"""#### BOXPLOT"""

fig, axs = plt.subplots(2,6, figsize=(20,6))

sns.boxplot(x1, ax=axs[0][0])
axs[0][0].set_title("Box plot of X1")

sns.boxplot(x2, ax=axs[0][1])
axs[0][1].set_title("Box plot of X2")

sns.boxplot(x3, ax=axs[0][2])
axs[0][2].set_title("Box plot of X3")

sns.boxplot(x4, ax=axs[0][3])
axs[0][3].set_title("Box plot of X4")

sns.boxplot(x5, ax=axs[0][4])
axs[0][4].set_title("Box plot of X5")

sns.boxplot(x6, ax=axs[0][5])
axs[0][5].set_title("Box plot of X6")

sns.boxplot(x7, ax=axs[1][0])
axs[1][0].set_title("Box plot of X7")

sns.boxplot(x8, ax=axs[1][1])
axs[1][1].set_title("Box plot of X8")

sns.boxplot(x9, ax=axs[1][2])
axs[1][2].set_title("Box plot of X9")

sns.boxplot(x10, ax=axs[1][3])
axs[1][3].set_title("Box plot of X10")

sns.boxplot(x11, ax=axs[1][4])
axs[1][4].set_title("Box plot of X11")

sns.boxplot(x12, ax=axs[1][5])
axs[1][5].set_title("Box plot of X12")

plt.tight_layout()
plt.show()

"""## DEPENDENT VARIABLE (Y) - AVG PRICE PER PLOT"""

y = data1["Avg Price per sqft (in Rs.)"]
y

"""#### DENSITY PLOT OF Y"""

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
sns.distplot(y)
axes.set_title("Density plot of Avg Price per sqft")

"""#### NORMALITY CHECKING USING SHAPIRO WILK TEST"""

stat1 , pvalue1 = scipy.stats.shapiro(y)
print(stat1)
print(pvalue1)

if pvalue1<=0.05:
    print("our data does not follow normal distribution")
else:
    print("our data follows normal distribution")

"""#### QQ PLOT"""

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
scipy.stats.probplot(y , dist = "norm" , plot = pylab)
plt.show()

"""#### BOXPLOT"""

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
sns.boxplot(x = y)
axes.set_title("Box plot of Avg Price of plot per sq feet")

"""## X-Y RELATIONSHIP"""

corr = data1.corr()
corr

sns.pairplot(corr)

fig2,axes2 = plt.subplots(figsize=(8, 7))
sns.heatmap(corr,annot=True, xticklabels= True , yticklabels= True , cmap="coolwarm")
axes2.set_title("Heatmap showing correlation between x and y values")

"""# LINEAR REGRESSION

## ORIGINAL DATA FITTING - APPROACH 1
"""

x_train1, x_test1, y_train1, y_test1 = train_test_split(x,y,test_size = 0.2, random_state = 3)

model_1 = LinearRegression()
model_1.fit(x_train1 , y_train1)

model_1.score(x_test1 , y_test1)

"""##### VISUALIZING THE MODEL"""

fig, axs = plt.subplots(2,6, figsize=(30,9))

sns.regplot(x=x1, y=y, data=data1, x_jitter=.05, ax=axs[0][0])
axs[0][0].set_title("Model x1 vs y")

sns.regplot(x=x2, y=y, data=data1, x_jitter=.05, ax=axs[0][1])
axs[0][1].set_title("Model x2 vs y")

sns.regplot(x=x3, y=y, data=data1, x_jitter=.05, ax=axs[0][2])
axs[0][2].set_title("Model x3 vs y")

sns.regplot(x=x4, y=y, data=data1, ax=axs[0][3])
axs[0][3].set_title("Model x4 vs y")

sns.regplot(x=x5, y=y, data=data1, x_jitter=.05, ax=axs[0][4])
axs[0][4].set_title("Model x5 vs y")

sns.regplot(x=x6, y=y, data=data1, x_jitter=.05, ax=axs[0][5])
axs[0][5].set_title("Model x6 vs y")

sns.regplot(x=x7, y=y, data=data1, x_jitter=.05, ax=axs[1][0])
axs[1][0].set_title("Model x7 vs y")

sns.regplot(x=x8, y=y, data=data1, x_jitter=.05, ax=axs[1][1])
axs[1][1].set_title("Model x8 vs y")

sns.regplot(x=x9, y=y, data=data1, x_jitter=.05, ax=axs[1][2])
axs[1][2].set_title("Model x9 vs y")

sns.regplot(x=x10, y=y, data=data1, x_jitter=.05, ax=axs[1][3])
axs[1][3].set_title("Model x10 vs y")

sns.regplot(x=x11, y=y, data=data1, x_jitter=.05, ax=axs[1][4])
axs[1][4].set_title("Model x11 vs y")

sns.regplot(x=x12, y=y, data=data1, x_jitter=.05, ax=axs[1][5])
axs[1][5].set_title("Model x12 vs y")

plt.tight_layout()

sns.jointplot(x=x1, y=y, data=data1, kind="reg")
sns.jointplot(x=x2, y=y, data=data1, kind="reg")
sns.jointplot(x=x3, y=y, data=data1, kind="reg")
sns.jointplot(x=x5, y=y, data=data1, kind="reg")
sns.jointplot(x=x6, y=y, data=data1, kind="reg")
sns.jointplot(x=x7, y=y, data=data1, kind="reg")
sns.jointplot(x=x8, y=y, data=data1, kind="reg")
sns.jointplot(x=x9, y=y, data=data1, kind="reg")
sns.jointplot(x=x10, y=y, data=data1, kind="reg")
sns.jointplot(x=x11, y=y, data=data1, kind="reg")
sns.jointplot(x=x12, y=y, data=data1, kind="reg")

"""##### R SQUARED"""

sklearn.metrics.r2_score(y_test1,y_pred_1)

"""##### Y TEST , PREDICTED Y """

y_pred_1 = model_1.predict(x_test1)

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
sns.distplot(y_test1)
axes.set_title("Density plot of the original y values")

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
sns.distplot(y_pred_1)
axes.set_title("Density plot of the predicted y values")

"""##### RESIDUALS"""

residuals_1 = y_test1-y_pred_1

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
sns.distplot(residuals_1)
axes.set_title("Density plot of the residual values")

"""##### Y , Y^ , RES"""

dict1 = {"y values" : y_test1 , "Predicted y (y^)" : y_pred_1 , "Residuals (y-y^)" : residuals_1}
df1 = pd.DataFrame(dict1)
print(df1)

"""##### 5 FOLD CROSS VALIDATION SCORE"""

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

cv_scores_1 =cross_val_score(model_1, x, y, cv=5)
print(cv_scores_1)

avg = np.mean(cv_scores_1)
std = np.std(cv_scores_1)
print("Average 5-Fold cross validation score - {}".format(avg))
print("Standard deviation - {}".format(std))

"""##### RMSE"""

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_1 = sqrt(mean_squared_error(y_test1, y_pred_1)) 
rmse_1

"""### ASSUMPTIONS CHECK

##### DATA DOES NOT FOLLOW NORMAL - ALREADY CHECKED THROUGH SHAPIRO WILK TEST , QQ PLOT AND BOX PLOT

##### HOMOSKEDASTICITY NOT TRUE

##### BP TEST
"""

import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

X = sm.add_constant(x_train1)
mod = sm.OLS(y_train1,X)
res = mod.fit()
res.summary()

#checking for heteroskedasticity
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(res.resid, X)

lzip(names, test)

import statsmodels
statsmodels.stats.diagnostic.het_goldfeldquandt(y_train1, x_train1)

"""##### PREDICTED Y VS RES"""

y1 = df1["Predicted y (y^)"]
e = df1["Residuals (y-y^)"]

plt.scatter(y1,e)
plt.axhline(y=0)
plt.xlabel("predicted y values")
plt.ylabel("residual values")
plt.title("Scatter plot of Predicted y vs Residuals")

"""##### MULTICOLLINEARITY EXISTS"""

from statsmodels.stats.outliers_influence import variance_inflation_factor

x
vif_data = pd.DataFrame()
vif_data["Independent Variables"] = x.columns

# calculating VIF for each independent variable
vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                          for i in range(len(x.columns))]
  
print(vif_data)

"""### APPROACH 2 - REMOVING X COLUMNS WITH HIGH VIF , KEEPING Y THE SAME -  REGRESSION FIT"""

x_new = x.drop(["No of health care centers(Hospitals/Pharmacy)","No of Educational Institutes",
                "No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)",
                "No of Restaurants", "No Religious Places(Temples/Mosques)", "No of Banks/ATMS"] , axis = 1)
y

x_train2, x_test2, y_train2, y_test2 = train_test_split(x_new,y,test_size = 0.1, random_state = 3)

model_2 = LinearRegression()
model_2.fit(x_train2 , y_train2)
model_2.score(x_test2 , y_test2)

"""##### R SQUARED"""

sklearn.metrics.r2_score(y_test2, y_pred_2)

"""##### Y TEST , PREDICTED Y """

y_pred_2 = model_2.predict(x_test2)
y_pred_2

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
sns.distplot(y_test2)
axes.set_title("Density plot of original y values")

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
sns.distplot(y_pred_2)
axes.set_title("Density plot of Predicted y values")

"""##### RESIDUALS"""

residuals_2 = y_test2-y_pred_2

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
sns.distplot(residuals_2)
axes.set_title("Density plot of the residual values")

"""##### Y , Y^ , RES"""

dict1 = {"y values" : y_test2 , "Predicted y (y^)" : y_pred_2 , "Residuals (y-y^)" : residuals_2}
df2 = pd.DataFrame(dict1)
print(df2)

"""##### 5 FOLD CROSS VALIDATION SCORE"""

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

cv_scores_2 =cross_val_score(model_2, x_new, y, cv=5)
print(cv_scores_2)

avg = np.mean(cv_scores_2)
std = np.std(cv_scores_2)
print("Average 5-Fold cross validation score - {}".format(avg))
print("Standard deviation - {}".format(std))

"""##### RMSE"""

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_2 = sqrt(mean_squared_error(y_test2, y_pred_2))
rmse_2

"""### TRYING TO MAKE THE SCORE BETTER

### APPROACH 3 - SCALED X(MINMAX SCALER) ; LOG(Y) -  REGRESSION FIT

##### MINMAX X
"""

x

scaler = preprocessing.RobustScaler()
robust_df = scaler.fit_transform(x)
robust_df = pd.DataFrame(robust_df , columns =["No. of metro stations","No. of railway stations(<12kms)",
                                               "No of bus stoppages","Distance from the Airport(kms)",
                                               "No of health care centers(Hospitals/Pharmacy)","No of Educational Institutes",
                                               "No of Departmental Stores","No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)",
                                               "No of Restaurants","No of Office",
                                               "No Religious Places(Temples/Mosques)","No of Banks/ATMS"])

scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(x)
standard_df = pd.DataFrame(standard_df , columns =["No. of metro stations","No. of railway stations(<12kms)",
                                               "No of bus stoppages","Distance from the Airport(kms)",
                                               "No of health care centers(Hospitals/Pharmacy)","No of Educational Institutes",
                                               "No of Departmental Stores","No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)",
                                               "No of Restaurants","No of Office",
                                               "No Religious Places(Temples/Mosques)","No of Banks/ATMS"])
 
scaler = preprocessing.MinMaxScaler()
minmax_df = scaler.fit_transform(x)
minmax_df = pd.DataFrame(minmax_df , columns =["No. of metro stations","No. of railway stations(<12kms)",
                                               "No of bus stoppages","Distance from the Airport(kms)",
                                               "No of health care centers(Hospitals/Pharmacy)","No of Educational Institutes",
                                               "No of Departmental Stores","No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)",
                                               "No of Restaurants","No of Office",
                                               "No Religious Places(Temples/Mosques)","No of Banks/ATMS"])
 
fig3,axes3 = plt.subplots(figsize=(20,6), nrows=1, ncols=4)

axes3[0].set_title('Before Scaling')
axes3[0].set_xlabel('X values')
sns.kdeplot(x["No. of metro stations"], ax = axes3[0], color ='red')
sns.kdeplot(x["No. of railway stations(<12kms)"], ax = axes3[0], color ='blue')
sns.kdeplot(x["No of bus stoppages"], ax = axes3[0], color ='green')
sns.kdeplot(x["Distance from the Airport(kms)"], ax = axes3[0], color ='yellow')
sns.kdeplot(x["No of health care centers(Hospitals/Pharmacy)"], ax = axes3[0], color ='purple')
sns.kdeplot(x["No of Educational Institutes"], ax = axes3[0], color ='black')
sns.kdeplot(x["No of Departmental Stores"], ax = axes3[0], color ='orange')
sns.kdeplot(x["No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)"], ax = axes3[0], color ='pink')
sns.kdeplot(x["No of Restaurants"], ax = axes3[0], color ='cyan')
sns.kdeplot(x["No of Office"], ax = axes3[0], color ='maroon')
sns.kdeplot(x["No Religious Places(Temples/Mosques)"], ax = axes3[0], color ='brown')
sns.kdeplot(x["No of Banks/ATMS"], ax = axes3[0], color ='olive')


axes3[1].set_title('After Robust Scaling')
axes3[1].set_xlabel('X values')
sns.kdeplot(robust_df["No. of metro stations"], ax = axes3[1], color ='red')
sns.kdeplot(robust_df["No. of railway stations(<12kms)"], ax = axes3[1], color ='blue')
sns.kdeplot(robust_df["No of bus stoppages"], ax = axes3[1], color ='green')
sns.kdeplot(robust_df["Distance from the Airport(kms)"], ax = axes3[1], color ='yellow')
sns.kdeplot(robust_df["No of health care centers(Hospitals/Pharmacy)"], ax = axes3[1], color ='purple')
sns.kdeplot(robust_df["No of Educational Institutes"], ax = axes3[1], color ='black')
sns.kdeplot(robust_df["No of Departmental Stores"], ax = axes3[1], color ='orange')
sns.kdeplot(robust_df["No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)"], ax = axes3[1], color ='pink')
sns.kdeplot(robust_df["No of Restaurants"], ax = axes3[1], color ='cyan')
sns.kdeplot(robust_df["No of Office"], ax = axes3[1], color ='maroon')
sns.kdeplot(robust_df["No Religious Places(Temples/Mosques)"], ax = axes3[1], color ='brown')
sns.kdeplot(robust_df["No of Banks/ATMS"], ax = axes3[1], color ='olive')

axes3[2].set_title('After Standard Scaling')
axes3[2].set_xlabel('X values')
sns.kdeplot(standard_df["No. of metro stations"], ax = axes3[2], color ='red')
sns.kdeplot(standard_df["No. of railway stations(<12kms)"], ax = axes3[2], color ='blue')
sns.kdeplot(standard_df["No of bus stoppages"], ax = axes3[2], color ='green')
sns.kdeplot(standard_df["Distance from the Airport(kms)"], ax = axes3[2], color ='yellow')
sns.kdeplot(standard_df["No of health care centers(Hospitals/Pharmacy)"], ax = axes3[2], color ='purple')
sns.kdeplot(standard_df["No of Educational Institutes"], ax = axes3[2], color ='black')
sns.kdeplot(standard_df["No of Departmental Stores"], ax = axes3[2], color ='orange')
sns.kdeplot(standard_df["No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)"], ax = axes3[2], color ='pink')
sns.kdeplot(standard_df["No of Restaurants"], ax = axes3[2], color ='cyan')
sns.kdeplot(standard_df["No of Office"], ax = axes3[2], color ='maroon')
sns.kdeplot(standard_df["No Religious Places(Temples/Mosques)"], ax = axes3[2], color ='brown')
sns.kdeplot(standard_df["No of Banks/ATMS"], ax = axes3[2], color ='olive')

axes3[3].set_title('After Min-Max Scaling')
axes3[3].set_xlabel('X values')
sns.kdeplot(minmax_df["No. of metro stations"], ax = axes3[3], color ='red')
sns.kdeplot(minmax_df["No. of railway stations(<12kms)"], ax = axes3[3], color ='blue')
sns.kdeplot(minmax_df["No of bus stoppages"], ax = axes3[3], color ='green')
sns.kdeplot(minmax_df["Distance from the Airport(kms)"], ax = axes3[3], color ='yellow')
sns.kdeplot(minmax_df["No of health care centers(Hospitals/Pharmacy)"], ax = axes3[3], color ='purple')
sns.kdeplot(minmax_df["No of Educational Institutes"], ax = axes3[3], color ='black')
sns.kdeplot(minmax_df["No of Departmental Stores"], ax = axes3[3], color ='orange')
sns.kdeplot(minmax_df["No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)"], ax = axes3[3], color ='pink')
sns.kdeplot(minmax_df["No of Restaurants"], ax = axes3[3], color ='cyan')
sns.kdeplot(minmax_df["No of Office"], ax = axes3[3], color ='maroon')
sns.kdeplot(minmax_df["No Religious Places(Temples/Mosques)"], ax = axes3[3], color ='brown')
sns.kdeplot(minmax_df["No of Banks/ATMS"], ax = axes3[3], color ='olive')

minmax_df

x1_m = minmax_df["No. of metro stations"]
x2_m = minmax_df["No. of railway stations(<12kms)"]
x3_m = minmax_df["No of bus stoppages"]
x4_m = minmax_df["Distance from the Airport(kms)"]
x5_m = minmax_df["No of health care centers(Hospitals/Pharmacy)"]
x6_m = minmax_df["No of Educational Institutes"]
x7_m = minmax_df["No of Departmental Stores"]
x8_m = minmax_df["No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)"]
x9_m = minmax_df["No of Restaurants"]
x10_m = minmax_df["No of Office"]
x11_m = minmax_df["No Religious Places(Temples/Mosques)"]
x12_m = minmax_df["No of Banks/ATMS"]

fig, axs = plt.subplots(2,6, figsize=(20,6))

sns.distplot(x1_m, ax=axs[0][0])
axs[0][0].set_title("Density plot of Scaled X1")

sns.distplot(x2_m, ax=axs[0][1])
axs[0][1].set_title("Density plot of Scaled X2")

sns.distplot(x3_m, ax=axs[0][2])
axs[0][2].set_title("Density plot of Scaled X3")

sns.distplot(x4_m, ax=axs[0][3])
axs[0][3].set_title("Density plot of Scaled X4")

sns.distplot(x5_m, ax=axs[0][4])
axs[0][4].set_title("Density plot of Scaled X5")

sns.distplot(x6_m, ax=axs[0][5])
axs[0][5].set_title("Density plot of Scaled X6")

sns.distplot(x7_m, ax=axs[1][0])
axs[1][0].set_title("Density plot of Scaled X7")

sns.distplot(x8_m, ax=axs[1][1])
axs[1][1].set_title("Density plot of Scaled X8")

sns.distplot(x9_m, ax=axs[1][2])
axs[1][2].set_title("Density plot of Scaled X9")

sns.distplot(x10_m, ax=axs[1][3])
axs[1][3].set_title("Density plot of Scaled X10")

sns.distplot(x11_m, ax=axs[1][4])
axs[1][4].set_title("Density plot of Scaled X11")

sns.distplot(x12_m, ax=axs[1][5])
axs[1][5].set_title("Density plot of Scaled X12")

plt.tight_layout()

"""##### LOG Y"""

y_log = np.log10(y)
y_log

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])

sns.distplot(y_log)
axes.set_title("Density plot of log y values")

"""##### MINMAX SCALED X & LOG Y -> ONE DATAFRAME"""

y_log_df = pd.DataFrame(y_log)
data2 = minmax_df.join(y_log_df , how = "outer")
data2

"""##### MODEL FITTING"""

x_train3, x_test3, y_train3, y_test3 = train_test_split(minmax_df,y_log_df,test_size = 0.1, random_state = 3)

model_3 = LinearRegression()
model_3.fit(x_train3 , y_train3)
model_3.score(x_test3 , y_test3)

y_pred3 = model_3.predict(x_test3)

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
sns.distplot(y_test3)
axes.set_title("Density plot of original y values")

fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])
sns.distplot(y_pred3)
axes.set_title("Density plot of the Predicted y values")

"""##### Y & Y^"""

index_ = y_test3.index
y_new = pd.DataFrame(y_pred3 , index_ , ["Predicted values"])

df4 = y_test3.join(y_new, how = "outer")
df4

"""##### RESIDUALS"""

yt = df4["Avg Price per sqft (in Rs.)"]
yp = df4["Predicted values"]

res = yt - yp
dict4 = {"Residuals":res}
df5 = pd.DataFrame(dict4)

df4.join(df5 , how="outer")

residuals = df5
fig = plt.figure(figsize = (5,5))
axes = fig.add_axes([0.1,0.1,0.9,0.9])

sns.distplot(residuals)
axes.set_title("Density plot of Residual values")

"""##### Y^ VS RES"""

plt.scatter(yp,res)
plt.axhline(y=0)
plt.xlabel("Predicted y values")
plt.ylabel("Residual values")
plt.title("Scatter plot of predicted values vs residuals")

"""##### RMSE"""

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse_3 = sqrt(mean_squared_error(yt,yp)) 
rmse_3

"""##### RMSE COMPARISON BETWEEN THE 3 APPROACHES"""

list1 = ["Model 1 - x & y" , "Model 2 - Large vif dropped x & y" , "Model 1 - Minmax scaled x & log y"]
list2 = [rmse_1 , rmse_2 , rmse_3]

dict2 = {"Model":list1 , "RMSE Values":list2}
pd.DataFrame(dict2)

"""# LOGISTIC REGRESSION

##### TURNING Y INTO A CATEGORICAL VARIABLE
"""

med_y = y.median()
med_y

y_list = []
for i in y:
    if i>med_y:
        i=1
    else:
        i=0
    y_list.append(i)
    
print(y_list)
len(y_list)

"""##### X & Y - DF"""

d = {"Avg Price per sqft in terms of HIGH and LOW price rating" : y_list}

y_list_df = pd.DataFrame(d)
y_list_df

DATA = x.join(y_list_df , how = "outer")
DATA

Y = DATA["Avg Price per sqft in terms of HIGH and LOW price rating"]

"""## PLOTTING FREQUENCY AND PERCENTAGE FOR HIGH AND LOW"""

a = Y.value_counts()
b = Y.value_counts(normalize = True)
print(b*100)

a.plot(kind = "bar")
plt.text(0,20,"50.22 %")
plt.text(1,20,"49.77 %")

"""## COVARIATES AS PER HIGH AND LOW PRICE"""

metro = pd.crosstab(x1,Y)
metro.plot.bar()

rail = pd.crosstab(x2,Y)
rail.plot.bar()

bus = pd.crosstab(x3,Y)
bus.plot.bar()

fig_1 = plt.figure(figsize=(15,5),dpi=100)

x4_1 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 1]["Distance from the Airport(kms)"]
x4_0 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 0]["Distance from the Airport(kms)"]


x4_1 = np.array(x4_1)
x4_0 = np.array(x4_0)

plt.hist([x4_1 , x4_0] , bins = 30 , label = ["Dist. from airport-high" , "Dist. from airport-low"])
plt.legend(loc = 1)

x4.max()
plt.xlim([0,250])
plt.xlabel("Distance from the airport as per High and Low price rate" )

fig_1 = plt.figure(figsize=(15,5),dpi=100)

x5_1 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 1]["No of health care centers(Hospitals/Pharmacy)"]
x5_0 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 0]["No of health care centers(Hospitals/Pharmacy)"]


x5_1 = np.array(x5_1)
x5_0 = np.array(x5_0)

plt.hist([x5_1 , x5_0] , bins = 30 , label = ["Hospitals-high" , "Hospitals-low"])
plt.legend(loc = 1)

x5.max()
plt.xlim([0,500])
plt.xlabel("Hospitals/Pharmacies as per High and Low price rate" )

fig_1 = plt.figure(figsize=(15,5),dpi=100)

x6_1 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 1]["No of Educational Institutes"]
x6_0 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 0]["No of Educational Institutes"]


x6_1 = np.array(x4_1)
x6_0 = np.array(x4_0)

plt.hist([x6_1 , x6_0] , bins = 5 , label = ["Edu In.-high" , "Edu In.-low"])
plt.legend(loc = 1)

x6.max()
plt.xlim([0,1850])
plt.xlabel("Educational Institutes as per High and Low price rate" )

fig_1 = plt.figure(figsize=(15,5),dpi=100)

x7_1 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 1]["No of Departmental Stores"]
x7_0 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 0]["No of Departmental Stores"]


x7_1 = np.array(x7_1)
x7_0 = np.array(x7_0)

plt.hist([x7_1 , x7_0] , bins = 30 , label = ["Stores-high" , "Stores-low"])
plt.legend(loc = 1)

x7.max()
plt.xlim([0,200])
plt.xlabel("Departmental Stores as per High and Low price rate" )

fig_1 = plt.figure(figsize=(15,5),dpi=100)

x8_1 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 1]["No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)"]
x8_0 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 0]["No of Recreational/ Entertainment Centres (Malls/Cinema Halls/Parks/Clubs)"]


x8_1 = np.array(x8_1)
x8_0 = np.array(x8_0)

plt.hist([x8_1 , x8_0] , bins = 30 , label = ["Rec places-high" , "Rec places-low"])
plt.legend(loc = 1)

x8.max()
plt.xlim([0,750])
plt.xlabel("Recreational places as per High and Low price rate" )

fig_1 = plt.figure(figsize=(15,5),dpi=100)

x9_1 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 1]["No of Restaurants"]
x9_0 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 0]["No of Restaurants"]


x9_1 = np.array(x9_1)
x9_0 = np.array(x9_0)

plt.hist([x9_1 , x9_0] , bins = 30 , label = ["Restaurants-high" , "Restaurants-low"])
plt.legend(loc = 1)

x9.max()
plt.xlim([0,1000])
plt.xlabel("Restaurants as per High and Low price rate" )

fig_1 = plt.figure(figsize=(15,5),dpi=100)

x10_1 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 1]["No of Office"]
x10_0 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 0]["No of Office"]


x10_1 = np.array(x10_1)
x10_0 = np.array(x10_0)

plt.hist([x10_1 , x10_0] , bins = 40 , label = ["Offices-high" , "Offices-low"])
plt.legend(loc = 1)

x10.max()
plt.xlim([0,350])
plt.xlabel("Offices as per High and Low price rate" )

fig_1 = plt.figure(figsize=(15,5),dpi=100)

x11_1 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 1]["No Religious Places(Temples/Mosques)"]
x11_0 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 0]["No Religious Places(Temples/Mosques)"]


x11_1 = np.array(x11_1)
x11_0 = np.array(x11_0)

plt.hist([x11_1 , x11_0] , bins = 30 , label = ["Religious places-high" , "Religious places-low"])
plt.legend(loc = 1)

x11.max()
plt.xlim([0,500])
plt.xlabel("Religious places as per High and Low price rate" )

fig_1 = plt.figure(figsize=(15,5),dpi=100)

x12_1 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 1]["No of Banks/ATMS"]
x12_0 = DATA[DATA["Avg Price per sqft in terms of HIGH and LOW price rating"] == 0]["No of Banks/ATMS"]


x12_1 = np.array(x12_1)
x12_0 = np.array(x12_0)

plt.hist([x12_1 , x12_0] , bins = 30 , label = ["Banks-high" , "Banks-low"])
plt.legend(loc = 1)

x12.max()
plt.xlim([0,450])
plt.xlabel("Banks as per High and Low price rate" )

"""##### CORRELATION BETWEEN X AND Y"""

corr = DATA.corr()

sns.pairplot(corr)

fig1,axes1 = plt.subplots(figsize=(8, 7))
sns.heatmap(corr, annot = True, xticklabels= True , yticklabels= True , cmap="coolwarm")

"""## BINARY CLASSIFICATION

## ON TESTING DATA
"""

x_train, x_test, y_train, y_test = train_test_split(x,y_list,test_size = 0.1, random_state = 3)

model_3 = LogisticRegression()
model_3.fit(x_train , y_train)
model_3.score(x_test , y_test)

"""##### VISUALIZING THE MODEL"""

fig, axs = plt.subplots(2,6, figsize=(30,9))

sns.regplot(x=x1, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[0][0])
axs[0][0].set_title("Model x1 vs y")

sns.regplot(x=x2, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[0][1])
axs[0][1].set_title("Model x2 vs y")

sns.regplot(x=x3, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[0][2])
axs[0][2].set_title("Model x3 vs y")

sns.regplot(x=x4, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[0][3])
axs[0][3].set_title("Model x4 vs y")

sns.regplot(x=x5, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[0][4])
axs[0][4].set_title("Model x5 vs y")

sns.regplot(x=x6, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[0][5])
axs[0][5].set_title("Model x6 vs y")

sns.regplot(x=x7, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[1][0])
axs[1][0].set_title("Model x7 vs y")

sns.regplot(x=x8, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[1][1])
axs[1][1].set_title("Model x8 vs y")

sns.regplot(x=x9, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[1][2])
axs[1][2].set_title("Model x9 vs y")

sns.regplot(x=x10, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[1][3])
axs[1][3].set_title("Model x10 vs y")

sns.regplot(x=x11, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[1][4])
axs[1][4].set_title("Model x11 vs y")

sns.regplot(x=x12, y=y_list, data=DATA, logistic=True, ci=None,ax=axs[1][5])
axs[1][5].set_title("Model x12 vs y")

plt.tight_layout()

"""##### INTERCEPT AND COEFFICIENTS"""

print("The intercept term is",model_3.intercept_)
print("The coefficients are ")
coeff = np.transpose(model_3.coef_)

pd.DataFrame(coeff , x.columns , ["Coefficients"])

"""##### PROBABILITIES OF Y PREDICTED"""

model_3.predict_proba(x_test)

y_pred = model_3.predict(x_test)

dict1 = {"Original y values (y)" : y_test , "Predicted y values (y^)" : y_pred}
df = pd.DataFrame(dict1)
df

"""##### 5 FOLD CROSS VALIDATION SCORE"""

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

cv_scores_1 =cross_val_score(model_3, x, y_list, cv=5)
print(cv_scores_1)

avg = np.mean(cv_scores_1)
std = np.std(cv_scores_1)
print("Average 5-Fold cross validation score - {}".format(avg))
print("Standard deviation - {}".format(std))

"""##### F1 SCORE"""

f1 = sklearn.metrics.f1_score(y_test, y_pred , average = "weighted")
f1

"""##### CONFUSION MATRIX , CLASSIFICATION REPORT"""

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

classification_report = classification_report(y_test, y_pred)
print(classification_report)

ConfMatrix = confusion_matrix(y_test,y_pred)
print(ConfMatrix)

"""##### PLOTTING CONFUSION MATRIX"""

sns.heatmap(ConfMatrix, annot = True , xticklabels = ["No default", "Default"], yticklabels = ["No default", "Default"])

plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion Matrix - Logistic Regression")

"""##### ACCURACY , PRECISION , RECALL"""

print("Accuracy - {}".format(metrics.accuracy_score(y_test, y_pred)))
print("Precision - {}".format(metrics.precision_score(y_test, y_pred,zero_division=1)))
print("Recall - {}".format(metrics.recall_score(y_test, y_pred,zero_division=0)))

"""##### ROC & AUC"""

y_pred_probability = model_3.predict_proba(x_test)
y_pred_probability = y_pred_probability[:,1]
y_pred_probability

auc_1 = metrics.roc_auc_score(y_test, y_pred_probability)
auc_1

"""##### ROC CURVE"""

fpr_1, tpr_1 , _ = metrics.roc_curve(y_test,  y_pred_probability)

plt.plot(fpr_1,tpr_1,label="auc = {}".format(auc_1))
plt.legend(loc=4)

"""## TRAINING DATA"""

x_train, x_test, y_train, y_test = train_test_split(x,y_list,test_size = 0.1, random_state = 3)

model_3 = LogisticRegression()
model_3.fit(x_train , y_train)
model_3.score(x_train , y_train)

"""##### INTERCEPT AND COEFFICIENTS"""

print("The intercept term is",model_3.intercept_)
print("The coefficients are ")
coeff = np.transpose(model_3.coef_)

pd.DataFrame(coeff , x.columns , ["Coefficients"])

"""##### PROBABILITY OF Y PREDICTED"""

model_3.predict_proba(x_train)

"""##### Y & Y^"""

y_pred_train = model_3.predict(x_train)

dict1 = {"Original y values (y)" : y_train , "Predicted y values (y^)" : y_pred_train}
df = pd.DataFrame(dict1)
df

"""##### 5 FOLD CROSS VALIDATION SCORE"""

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

cv_scores_1 =cross_val_score(model_3, x, y_list, cv=5)
print(cv_scores_1)

avg = np.mean(cv_scores_1)
std = np.std(cv_scores_1)
print("Average 5-Fold cross validation score - {}".format(avg))
print("Standard deviation - {}".format(std))

"""##### F1 SCORE"""

sklearn.metrics.f1_score(y_train, y_pred_train , average = "weighted")

"""##### CLASSIFICATION REPORT AND CONFUSION MATRIX"""

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

classification_report = classification_report(y_train, y_pred_train)
print(classification_report)

ConfMatrix = confusion_matrix(y_train, y_pred_train)
print(ConfMatrix)

"""##### PLOTTING CONFUSION MATRIX"""

sns.heatmap(ConfMatrix, annot = True , xticklabels = ["No default", "Default"], yticklabels = ["No default", "Default"])

plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion Matrix - Logistic Regression")

"""##### ACCURACY , PRECISION , RECALL """

print("Accuracy - {}".format(metrics.accuracy_score(y_train, y_pred_train)))
print("Precision - {}".format(metrics.precision_score(y_train, y_pred_train)))
print("Recall - {}".format(metrics.recall_score(y_train, y_pred_train)))

"""##### ROC & AUC"""

y_pred_probability_train = model_3.predict_proba(x_train)
y_pred_probability_train = y_pred_probability_train[:,1]
y_pred_probability_train

auc_1_train = metrics.roc_auc_score(y_train, y_pred_train)
auc_1_train

"""##### ROC CURVE"""

fpr_1_train, tpr_1_train , _ = metrics.roc_curve(y_train,  y_pred_probability_train)

plt.plot(fpr_1_train, tpr_1_train,label="auc = {}".format(auc_1_train))
plt.legend(loc=4)

"""##### ROC CURVE FOR TESTING AND TRAINING DATASET"""

fig_1 = plt.figure(figsize=(5,5),dpi=100)
axes_1 = fig_1.add_axes([0.1,0.1,0.9,0.9])


axes_1.set_title("Roc curve for training and testing data")

axes_1.plot(fpr_1,tpr_1,label="auc = {}".format(auc_1)) 

axes_1.plot(fpr_1_train, tpr_1_train,label="auc = {}".format(auc_1_train))         
axes_1.legend(loc=4)

"""## EXTRATREESCLASSIFIER"""

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(x,y_list)

#use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=x.columns)
print(feat_importances.sort_values(ascending=False).cumsum())
feat_importances.nlargest(12).plot(kind='barh')
plt.show()

"""## MULTICLASS"""

y

q1 = np.percentile(y,25)
q2 = np.percentile(y,50)
q3 = np.percentile(y,75)

print("The quantiles of y are ->")
print("Q1 : {} , Q2 : {} , Q3 : {}".format(q1,q2,q3))

y_list2 = []
for i in y:
    if i<=q1:
        i=0
    elif (i>q1) and (i<q3):
        i=1
    else:
        i=2
    y_list2.append(i)
    
print(y_list2)
len(y_list2)

"""### ON TESTING DATA"""

x_train, x_test, y_train, y_test = train_test_split(x,y_list2,test_size = 0.1, random_state = 1)

model_4 = LogisticRegression(multi_class="multinomial",solver="lbfgs")
model_4.fit(x_train , y_train)
model_4.score(x_test , y_test)

"""##### INTERCEPT AND COEFFICIENTS"""

print("The intercept term is",model_4.intercept_)
print("The coefficients are ")

coeff2 = np.transpose(model_4.coef_)

pd.DataFrame(coeff2 , x.columns , ["For class 0", "For class 1", "For class 2"])

"""##### Y , Y^"""

y_pred = model_4.predict(x_test)

dict1 = {"Original y values (y)" : y_test , "Predicted y values (y^)" : y_pred}
df = pd.DataFrame(dict1)
df

"""##### 5 FOLD CROSS VALIDATION SCORE"""

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

cv_scores_1 =cross_val_score(model_4, x, y_list, cv=5)
print(cv_scores_1)

avg = np.mean(cv_scores_1)
std = np.std(cv_scores_1)
print("Average 5-Fold cross validation score - {}".format(avg))
print("Standard deviation - {}".format(std))

"""##### F1 SCORE"""

sklearn.metrics.f1_score(y_test, y_pred , average = "weighted")

"""##### CLASSFICATION REPORT & CONFUSION MATRIX"""

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

classification_report = classification_report(y_test, y_pred)
print(classification_report)

ConfMatrix = confusion_matrix(y_test,y_pred)
print(ConfMatrix)

"""##### PLOTTING THE CONFUSION MATRIX"""

sns.heatmap(ConfMatrix, annot = True , xticklabels = ["No default", "Default"], yticklabels = ["No default", "Default"])

plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion Matrix - Logistic Regression")

"""##### ACCURACY , PRECISION , RECALL"""

print("Accuracy - {}".format(metrics.accuracy_score(y_test, y_pred)))
print("Precision - {}".format(metrics.precision_score(y_test, y_pred , average = "weighted")))
print("Recall - {}".format(metrics.recall_score(y_test, y_pred , average = "weighted")))

"""##### ROC"""

y_pred_probability = model_4.predict_proba(x_test)
y_pred_probability

"""##### AUC"""

auc_1 = metrics.roc_auc_score(y_test, y_pred_probability,multi_class = "ovo")
auc_1

"""### ON TRAINING DATA"""

x_train, x_test, y_train, y_test = train_test_split(x,y_list2,test_size = 0.1, random_state = 1)

model_4 = LogisticRegression(multi_class="multinomial",solver="lbfgs")
model_4.fit(x_train , y_train)
model_4.score(x_train , y_train)

"""##### INTERCEPT AND COEFFICIENTS"""

print("The intercept term is",model_4.intercept_)
print("The coefficients are ")

coeff2 = np.transpose(model_4.coef_)

pd.DataFrame(coeff2 , x.columns , ["For class 0", "For class 1", "For class 2"])

"""##### Y & Y^"""

y_pred_train = model_4.predict(x_train)

dict1 = {"Original y values (y)" : y_train , "Predicted y values (y^)" : y_pred_train}
df = pd.DataFrame(dict1)
df

"""##### 5 FOLD CROSS VALIDATION SCORE"""

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

cv_scores_1 =cross_val_score(model_4, x, y_list, cv=5)
print(cv_scores_1)

avg = np.mean(cv_scores_1)
std = np.std(cv_scores_1)
print("Average 5-Fold cross validation score - {}".format(avg))
print("Standard deviation - {}".format(std))

"""##### F1 SCORE"""

sklearn.metrics.f1_score(y_train, y_pred_train , average = "weighted")

"""##### CLASSIFICATION REPORT AND CONFUSION MATRIX"""

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

classification_report = classification_report(y_train, y_pred_train)
print(classification_report)

ConfMatrix = confusion_matrix(y_train, y_pred_train)
print(ConfMatrix)

"""##### PLOTTING THE CONFUSION MATRIX"""

sns.heatmap(ConfMatrix, annot = True , xticklabels = ["No default", "Default"], yticklabels = ["No default", "Default"])

plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion Matrix - Logistic Regression")

"""##### ACCURACY, PRECISION, RECALL"""

print("Accuracy - {}".format(metrics.accuracy_score(y_train, y_pred_train)))
print("Precision - {}".format(metrics.precision_score(y_train, y_pred_train , average = "weighted")))
print("Recall - {}".format(metrics.recall_score(y_train, y_pred_train , average = "weighted")))

"""##### ROC"""

y_pred_probability_train = model_4.predict_proba(x_train)
y_pred_probability_train

"""##### AUC"""

auc_1 = metrics.roc_auc_score(y_train, y_pred_probability_train,multi_class = "ovo")
auc_1



