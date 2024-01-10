# %%
from functions import Dataset

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

import pandas as pd
import polars as pl
import polars.selectors as cs
import random
import os

import numpy as np

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score, cross_val_predict, TimeSeriesSplit
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor

from functions import Dataset

def cv_(sets_):
    ids_ = sets_.with_row_count()
    yield ids_.filter(pl.col('data_block_id').is_in(dataset.train_id))['row_nr'].to_numpy(),\
        ids_.filter(pl.col('data_block_id').is_in(dataset.test_id))['row_nr'].to_numpy()

# Data
dataset = Dataset()
dataset.load()

# Sets
sets_  =  dataset.merge_set()

# Estimation
features_prod  = ~cs.matches('.*(datetime|_id|latitude|longitude|is_consumption|hours_ahead|target).*')
categories = sets_.select(~features_prod & cs.integer()).columns

#
model_parameters = {"n_estimators": 10,"objective": "regression_l1","learning_rate": 0.05,"colsample_bytree": 0.89,"colsample_bynode": 0.596,"lambda_l1": 3.4895,"lambda_l2": 1.489,"max_depth": 15,"num_leaves": 490,"min_data_in_leaf": 48,'max_bin':840, 'force_col_wise':True}

class Model():
    def __init__(self, model_parameters, features_pred_prod, features_pred_conso, n_model=3):
        self.features_pred_prod, self.features_pred_conso = features_pred_prod, features_pred_conso

        self.model_production = VotingRegressor(
                                [(f'production_{i}', LGBMRegressor(**model_parameters, random_state=i)) for i in range(n_model)]
                            )
        self.model_consomation = VotingRegressor(
                                [(f'consomation_{i}', LGBMRegressor(**model_parameters, random_state=i)) for i in range(n_model)]
                            )
#
    def fit(self, sets_, features_prod, features_cons):
        X = sets_.filter(sets_['is_consumption']==0)
        self.model_production.fit(X.select(features_prod), X['target'])

        X = sets_.filter(sets_['is_consumption']==1)
        self.model_consomation.fit(X.select(features_cons), X['target'])

    def predict(self, sets_, ):

        predictions = np.zeros(len(sets_),)

        X = sets_.filter(sets_['is_consumption']==0)
        predictions[sets_['is_consumption']==0] = self.model_production.predict(X.select(self.features_pred_prod))

        X = sets_.filter(sets_['is_consumption']==1)
        predictions[sets_['is_consumption']==1] = self.model_consomation.predict(X.select(self.features_pred_conso))

        return predictions.clip(0)

#
model = Model(model_parameters)
model.fit(sets_, features_prod, features_prod)



# #
# scores_prods = cross_val_score(model, X.select(features_prod), X['target'], cv=cv_(X), scoring='neg_mean_absolute_error',error_score='raise')

# scores_prods.mean()

# scores_prods = cross_val_score(model_production, X.select(features_prod), X['target'], cv=cv_(X), scoring='neg_mean_absolute_error',error_score='raise')

# scores_prods.mean()

# %%
import enefit

env = enefit.make_env()
iter_test = env.iter_test()

for (df_test, df_new_target, df_new_client, df_new_historical_weather,df_new_forecast_weather, df_new_electricity_prices, df_new_gas_prices, df_sample_prediction) in iter_test:

    dataset.update_data(
        df_new_client=df_new_client,
        df_new_gas_prices=df_new_gas_prices,
        df_new_electricity_prices=df_new_electricity_prices,
        df_new_forecast_weather=df_new_forecast_weather,
        df_new_historical_weather=df_new_historical_weather,
        df_new_target=df_new_target
    )
    df_test_features = dataset.merge_set()
    
    df_sample_prediction["target"] = model.predict(df_test_features[df_test['row_id'].values], features_prod, features_prod)
    
    env.predict(df_sample_prediction)
