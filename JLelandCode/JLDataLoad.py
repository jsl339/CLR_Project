## writing a data loader for our three datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch, torchvision
## imports go above here
## load data
from ucimlrepo import fetch_ucirepo 
import numpy as np
# fetch dataset 

def data_cleaner(name):

    if name == "adult":
        data = fetch_ucirepo(name = name) 
        # load data
        df = pd.read_csv(data.metadata['data_url'])

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    elif name == "Breast Cancer Wisconsin":
        
        data = fetch_ucirepo(name = name) 
        # load data
        df = pd.read_csv(data.metadata['data_url'])
        df.dropna(inplace=True)
        ## Y's are the class
        ## X's are really anything else, minus sample code number
        X = df.drop(["Class"], axis=1)
        X.drop(["Sample_code_number"], axis=1, inplace=True)
        y = df["Class"]

        ## reclass y into 0,1 instead of 2,4
        y = y.replace(2, 0)
        y = y.replace(4, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    elif name == "fmnist":
        train_set = torchvision.datasets.FashionMNIST("/Users/johnleland/Downloads/", download=True)
        test_set = torchvision.datasets.FashionMNIST("/Users/johnleland/Downloads/",download=True,train=False)
        X_train = train_set.data.numpy()
        labels_train = train_set.targets.numpy()
        X_test = test_set.data.numpy()
        labels_test = test_set.targets.numpy()
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
        X_train = X_train/255.0
        X_test = X_test/255.0
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.DataFrame(labels_train)
        y_test = pd.DataFrame(labels_test)

    return X_train, X_test, y_train, y_test