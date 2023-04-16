# -*- coding: utf-8 -*-
"""crop price prediction model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1havgZYtvHSyG7togo1t_UcKCXH9Z6J3m
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.ensemble import RandomForestRegressor
# %matplotlib inline

def predict(commodity, state, district, market):
    df = pd.read_csv('crop price.csv',na_values='+')
    df.shape

    df.info()

    df['state'].unique()

    df.isnull().sum()

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(df['state'],df['modal_price'])
    plt.show()

    df2=df.copy()

    dist=(df2['commodity'])
    distset=set(dist)
    dd=list(distset)
    dictOfWords={dd[i]: i for i in range (0, len(dd))}
    df2['commodity']=df2['commodity'].map(dictOfWords)

    dist=(df2['state'])
    distset=set(dist)
    dd=list(distset)
    dictOfWords={dd[i]: i for i in range (0, len(dd))}
    df2['state']=df2['state'].map(dictOfWords)

    dist=(df2['district'])
    distset=set(dist)
    dd=list(distset)
    dictOfWords={dd[i]: i for i in range (0, len(dd))}
    df2['district']=df2['district'].map(dictOfWords)

    dist=(df2['market'])
    distset=set(dist)
    dd=list(distset)
    dictOfWords={dd[i]: i for i in range (0, len(dd))}
    df2['market']=df2['market'].map(dictOfWords)

    df.head()

    df2.head()
    df2.columns

    features = df2[['commodity', 'state', 'district', 'market']]
    labels = df2['modal_price']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=10)

    model = RandomForestRegressor(max_depth = 1000, random_state=0)
    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    r2_score(y_test, y_pred)

    user_input = [[271, 1, 3, 5]]
    model.predict(user_input)

    X1_test=np.array([commodity, state, district, market])
    X1_test=X1_test.reshape((1,-1))
    return model.predict(X1_test)[0]