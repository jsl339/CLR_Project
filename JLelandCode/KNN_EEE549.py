import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from JLDataLoad import data_cleaner
## imports go above here

## load data
from ucimlrepo import fetch_ucirepo 
"""# fetch dataset 
adult = fetch_ucirepo(name = 'adult') 

## this is how you pull the dataset according to UCI
# data (as pandas dataframes) 
df = pd.read_csv(adult.metadata['data_url'])

df.replace('?', np.nan, inplace=True)

X = df.drop(["income"], axis=1)
y = df["income"]


## there are technically 4 different y's. <=50k, >50k, <=50k., >50k.
## this is the stupidest possible way to fix it. But it works

y = y.replace("<=50K", 0)
y = y.replace("<=50K.", 0)
y = y.replace(">50K.", 1)
y = y.replace(">50K", 1)

# drop useless columns
X.drop(["fnlwgt","education-num"], axis=1, inplace=True)

## categorical variables need to be dealt with
## convert from object to category
# this is to stop the warnings that I'm using a slice of a dataframe when setting the cat values
pd.options.mode.chained_assignment = None  # default='warn'
for i in ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]:
    X[i] = X[i].astype('category')
    X[i] = X[i].cat.codes
# scale non-categorical variables

scaler = StandardScaler()
for i in ["age","capital-gain","capital-loss","hours-per-week"]:
    X[i] = scaler.fit_transform(X[[i]])

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)"""

## now need to set up cross validation

## TO DO: Need to do cross validation on the number of components as well as the number of neightbors. 
## grid search?

X_train, X_test, y_train, y_test = data_cleaner("fmnist")


big_fucker = []

cv = KFold(n_splits=10, random_state=42, shuffle=True)
component_list = [10]
for j in  component_list:
    print(j)
    knn = KNeighborsClassifier(n_neighbors=j, n_jobs=-1) # more cores go BRRRRRT
    accounting = []
    for train_index, val_index in cv.split(X_train):
        pca = PCA(n_components=10, random_state=42)
        expl_train, expl_val = X_train.iloc[train_index], X_train.iloc[val_index]
        response_train, response_val = y_train.iloc[train_index], y_train.iloc[val_index]

        ## pca
        expl_pca_train = pca.fit_transform(expl_train)
        expl_pca_val = pca.transform(expl_val)

        ## fit
        knn.fit((expl_pca_train), response_train.values.ravel())

        ## score
        predict = knn.predict(expl_pca_val)
        score = np.mean(predict == response_val.values.ravel())
        accounting.append(score)
        print(score)
    big_fucker.append((np.mean(accounting)))

#plt.plot(component_list, big_fucker)
#plt.show()