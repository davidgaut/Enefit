# %%
# import os
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# import seaborn as sns
# import pandas as pd

import random
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import polars as pl
import polars.selectors as cs

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, TimeSeriesSplit
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor

from functions import Dataset

# %%
# [√] Normalize production / capacity and clip to 1
# [ ] County for weather?
# Contract type and energy prices
# Add frequencies
# Linear Trees
# Hyperparams (production / consumption)

# %%
# Data
dataset = Dataset()
dataset.load()

# Sets
# info , client, g_price, e_price, f_weather, h_weather, _ ,loc_stats, loc_stats_micro = dataset.info,dataset.client,dataset.g_price,dataset.e_price,dataset.f_weather,dataset.h_weather,dataset.ws_county, dataset.loc_stats, dataset.loc_stats_micro
sets_, dates = dataset.make_features(dataset.info,dataset.client,dataset.g_price,dataset.e_price,dataset.f_weather,dataset.h_weather, dataset.loc_stats,dataset.loc_stats_micro)

# Missing
sets_.with_columns(pl.all().is_null()).sum().transpose(include_header=True).filter(pl.col('column_0')!=0)

# %%
# Estimation
features_prod = sets_.select( ~cs.matches('.*(datetime|_id|latitude|longitude|is_consumption|hours_ahead|target).*')).columns
features_cons = sets_.select( ~cs.matches('.*(datetime|_id|latitude|longitude|is_consumption|hours_ahead|target).*')).columns

#
model_parameters = {"n_estimators": 500,"objective": "regression_l1","learning_rate": 0.05,"colsample_bytree": 0.89,"colsample_bynode": 0.596,"lambda_l1": 3.4895,"lambda_l2": 1.489,"max_depth": 15,"num_leaves": 490,"min_data_in_leaf": 48,'max_bin':840, 'force_col_wise':True, 'n_jobs':-1}

class Model(BaseEstimator,RegressorMixin):
    def __init__(self, n_estimators=100, objective='regression_l1',path_smooth=0.0,learning_rate=0.1,colsample_bytree=1,colsample_bynode=0.5,lambda_l1=0.0,lambda_l2=0.0,max_depth=-1,num_leaves=31,min_data_in_leaf=20,max_bin=100,force_col_wise=True, features_pred_prod=None, features_pred_cons=None, n_model=3,n_jobs=-1):

        model_parameters = {"n_estimators":      n_estimators,
                            "path_smooth":      path_smooth,
                            "objective":         objective,
                            "learning_rate":     learning_rate,
                            "colsample_bytree":  colsample_bytree,
                            "colsample_bynode":  colsample_bynode,
                            "lambda_l1":         lambda_l1,
                            "lambda_l2":         lambda_l2,
                            "max_depth":         max_depth,
                            "num_leaves":        num_leaves,
                            "min_data_in_leaf":  min_data_in_leaf,
                            'max_bin':           max_bin,
                            'force_col_wise':    force_col_wise,
                            }
        
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.path_smooth = path_smooth
        self.objective = objective
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.max_bin = max_bin
        self.force_col_wise = force_col_wise

        self.model_parameters = model_parameters
        self.n_model = n_model
        self.features_pred_prod, self.features_pred_cons = features_pred_prod, features_pred_cons

        self.model_prod = VotingRegressor(
                                [(f'production_{i}', LGBMRegressor(**model_parameters, random_state=i)) for i in range(self.n_model)]
                            )
        self.model_cons = VotingRegressor(
                                [(f'consomation_{i}', LGBMRegressor(**model_parameters, random_state=i)) for i in range(self.n_model)]
                            )
        #
        self.fit_prod = list(tuple(zip(self.model_prod.named_estimators.keys(), [[{'categorical_features':sets_[features_prod].select(cs.integer()).columns}]] *self.n_model)))
        self.fit_cons = list(tuple(zip(self.model_cons.named_estimators.keys(), [[{'categorical_features':sets_[features_cons].select(cs.integer()).columns}]] *self.n_model)))

    def fit(self, X, y):
        is_prod = X.with_row_count().filter(pl.col('is_consumption')==0).select(pl.col('row_nr')).to_numpy().flatten()
        is_cons = X.with_row_count().filter(pl.col('is_consumption')==1).select(pl.col('row_nr')).to_numpy().flatten()
        
        cat_cols = sets_[features_prod].select(cs.integer() | cs.string()).columns
        X_prod, y_prod = X[is_prod].select(self.features_pred_prod).to_pandas(), y[is_prod].to_pandas()
        X_prod[cat_cols] = X_prod[cat_cols].astype("category")
        self.model_prod.fit(X_prod,y_prod)

        cat_cols = sets_[features_cons].select(cs.integer() | cs.string()).columns
        X_cons, y_cons = X[is_cons].select(self.features_pred_cons).to_pandas(), y[is_cons].to_pandas()
        X_cons[cat_cols] = X_cons[cat_cols].astype("category")
        self.model_cons.fit(X_cons,y_cons)

        return self

    def predict(self, X):
        is_prod = X.with_row_count().filter(pl.col('is_consumption')==0).select(pl.col('row_nr')).to_numpy().flatten()
        is_cons = X.with_row_count().filter(pl.col('is_consumption')==1).select(pl.col('row_nr')).to_numpy().flatten()
        
        predictions = np.zeros(len(X),)

        cat_cols = sets_[features_prod].select(cs.integer() | cs.string()).columns
        X_prod = X[is_prod].select(self.features_pred_prod).to_pandas()
        X_prod[cat_cols] = X_prod[cat_cols].astype("category")
        predictions[is_prod] = self.model_prod.predict(X_prod)

        cat_cols = sets_[features_cons].select(cs.integer() | cs.string()).columns
        X_cons = X[is_cons].select(self.features_pred_cons).to_pandas()
        X_cons[cat_cols] = X_cons[cat_cols].astype("category")       
        predictions[is_cons] = self.model_cons.predict(X_cons)
        
        predictions = predictions.clip(0, 1) * X['installed_capacity']

        return predictions.clip(0)

# model = Model(**model_parameters,features_pred_prod=features_prod, features_pred_cons=features_cons)

# model.fit(sets_, sets_['target'])
# y_pred = model.predict(sets_)


# y_pred_cross = cross_validate(model, sets_, sets_['target'], cv=cv_split)
# mae(sets_['target']*sets_['installed_capacity'],y_pred_cross)

# Time Series CV
scores = list()
cv_split = TimeSeriesSplit(n_splits=5).split(sets_['target'])
for train_cv_id, test_cv_id in cv_split:
    X_cv_train, X_cv_test = sets_[train_cv_id], sets_[test_cv_id]
    model = Model(**model_parameters, features_pred_prod=features_prod, features_pred_cons=features_cons)
    model.fit(X_cv_train, X_cv_train['target'])
    y_pred = model.predict(X_cv_test)
    scores.append(mae(X_cv_test['target']*X_cv_test['installed_capacity'],y_pred))

print(np.mean(scores))

# %%

if False:
    def tune_lgbm_model(X_train, y_train, random_state, n_iter=5, cv=3):
        param_dist = {
            'path_smooth': sp_uniform(0.01, 0.09),
            'colsample_bytree': sp_uniform(0.1, 0.9),
            'colsample_bynode': sp_uniform(0.1, 0.9),
            'lambda_l1': sp_uniform(1, 9), 
            'lambda_l2': sp_uniform(1, 9), 
            'num_leaves': sp_randint(31, 500),
            'min_data_in_leaf': sp_randint(30, 250),
            'max_depth': sp_randint(1, 29),
            'learning_rate': sp_uniform(0.001, 0.09),
            'objective': 'regression_l1',#['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie']
        }

        lgb_reg = lgb.LGBMRegressor(num_iterations=1,verbose=-1)

        random_search = RandomizedSearchCV(
            estimator=lgb_reg, 
            param_distributions=param_dist,
            n_iter=n_iter, 
            scoring='neg_mean_absolute_error',
            cv=TimeSeriesSplit(cv),
            error_score='raise',
            verbose=0, 
            random_state=random_state
        )

        random_search.fit(X_train, y_train)

        return random_search.best_params_

    random_states = random.sample(range(1, 100), 10)

    # Consumption
    ii = 0
    parameter_list_conso = []
    id_consumption = sets_.with_row_count().filter(pl.col('is_consumption')==1)['row_nr']
    for rs in random_states:
        print(ii)
        ii+=1
        best_params = tune_lgbm_model(sets_[id_consumption], sets_['target'][id_consumption], rs)
        best_params_with_fixed = {
            'num_iterations': 100,
            'n_jobs':-1,
            'verbose': -1,
            **best_params
        }
        parameter_list_conso.append(best_params_with_fixed)
        print(f"Best parameters for random state {rs}: {best_params_with_fixed}")

    # Production
    ii = 0
    id_production = sets_.with_row_count().filter(pl.col('is_consumption')==0)['row_nr']
    parameter_list_prod = []
    for rs in random_states:
        print(ii)
        ii+=1
        best_params = tune_lgbm_model(sets_, sets_['target'], rs)
        best_params_with_fixed = {
            'num_iterations': 100,
            'n_jobs':-1,
            'verbose': -1,
            **best_params
        }
        parameter_list_prod.append(best_params_with_fixed)
        print(f"Best parameters for random state {rs}: {best_params_with_fixed}")


# %%
import enefit
env = enefit.make_env()
iter_test = env.iter_test()

last_data_block_id = dataset.blocks.max()

for data_block_id, (df_test, df_new_target, df_new_client, df_new_historical_weather,df_new_forecast_weather, df_new_electricity_prices, df_new_gas_prices, df_sample_prediction) in enumerate(iter_test):
    
    dataset.update_data(
        data_block_id= 1 + data_block_id + last_data_block_id,
        df_new_client=df_new_client,
        df_new_gas_prices=df_new_gas_prices,
        df_new_electricity_prices=df_new_electricity_prices,
        df_new_forecast_weather=df_new_forecast_weather,
        df_new_historical_weather=df_new_historical_weather,
        df_new_target=df_new_target
    )
    info,client,g_price,e_price,f_weather,h_weather,loc_stats,loc_stats_micro = dataset.info,dataset.client,dataset.g_price,dataset.e_price,dataset.f_weather,dataset.h_weather,dataset.loc_stats,dataset.loc_stats_micro
    df_test_features, dates = dataset.make_features(info,client,g_price,e_price,f_weather,h_weather,loc_stats,loc_stats_micro)

    df_sample_prediction["target"] = model.predict(df_test_features[df_test['row_id'].values])
    
    env.predict(df_sample_prediction)
