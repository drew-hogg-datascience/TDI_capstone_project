import numpy as np
import pandas as pd
#from __future__ import absolute_import
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from DL_setup import *

def filter_data(data):

    data = data[FIELDS['all']].dropna(how='any')

    # Make column for historically marginalized groups
    data['PCT_HISTMARG'] = (data['PCT_NHBLACK10'] +
                            data['PCT_HISP10'] +
                            data['PCT_NHNA10'] +
                            data['PCT_NHPI10'])

    # Make column for an estimate of households w/o car
    data['PCT_NOCAR'] = data['PCT_LACCESS_HHNV15'] - data['PCT_LACCESS_POP15']

    return data

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(28, input_dim=28, kernel_initializer='normal', activation='relu'))
    model.add(Dense(28, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def DL_model(full_data):

    # Filter data and get predictors
    data = filter_data(full_data)

    predictors = data.columns.tolist()

    predictors = [p for p in predictors if p not in FIELDS['exclude']]

    print 'Total predictors: ', len(predictors)
    neurons = len(predictors)

    # Separate data
    X = data[predictors]
    Y = data[target_field]

    # Random seed
    seed = 0
    np.random.seed(seed)

    # Define estimator and run model with cross validation
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold)

    # Print results
    print 'Individual cross validation results: ', results
    print('Results: %.2f (%.2f) MSE' % (results.mean(), results.std()))
