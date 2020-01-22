import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import os
print(os.getcwd())

os.chdir('E:\\Locker\\Sai\\SaiHCourseNait\\DecBtch\\R_Datasets\\')
print(os.getcwd())

data = pd.read_csv('diabetes2.csv')

data.info()

%matplotlib inline


import matplotlib.pyplot as plt
import seaborn as sns
import re
from IPython.display import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor,
RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call



array = data.values

array

type(array)

X = array[:,0:8] 
X

y = array[:,8] 
y

test_size = 0.33
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size)
print('Partitioning Done!')

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

model = DecisionTreeClassifier()

model.fit(X_train,y_train)

prediction = model.predict(X_test)
prediction

outcome = y_test
outcome

print(metrics.accuracy_score(outcome,prediction))

print(metrics.confusion_matrix(y_test,prediction)) #

model.feature_importances_

expected = y_test
predicted = prediction
conf = metrics.confusion_matrix(expected, predicted)
print(conf)

label = ["0","1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)

importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
feature = data[data.columns[0:8]]
feat_names = data.columns[0:8]
print("DecisionTree Feature ranking:")
for f in range(feature.shape[1]):
print("%d. feature %s (%f)" % (f + 1, feat_names[indices[f]], importance[indices[f]]))
plt.figure(figsize=(15,5))
plt.title("DecisionTree Feature importances")
plt.bar(range(feature.shape[1]), importance[indices], color="y", align="center")
plt.xticks(range(feature.shape[1]), feat_names[indices])
plt.xlim([-1, feature.shape[1]])
plt.show()

print(metrics.accuracy_score(outcome,prediction))