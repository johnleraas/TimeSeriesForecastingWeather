# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 08:25:18 2022

"""
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns


def add_two(x):
    return x+2

def divide_three(x):
    return x/3

# Ensure proper format
def ensure_categorical(df, col_list):
  for col in col_list:
    df[col]=df[col].astype('category')
  return df

# Load File
def load_file(file):
  df = pd.read_csv(file, index_col=0)

  # Ensure Date proper format
  df['Date']=pd.to_datetime(df['Date'])

  df = df.set_index('Date')

  # Select Categorical Columns and Assign
  cat_cols = ['dow', 'fed_holiday', 'is_wknd', 'is_month_start', 'is_month_end', 'wkday_fedholiday_lag1']
  df = ensure_categorical(df=df, col_list=cat_cols)

  # Select all "Holiday Lag" columns
  cat_cols = [col for col in df.columns.to_list() if col.startswith('fed_holiday_lag')]
  df = ensure_categorical(df=df, col_list=cat_cols)

  return df

def train_test_split_ts(df, date_val):
  # Reset Index
  df.reset_index(inplace=True)

  # Get Index
  date_index = df.index[df.Date==date_val][0]

  # Assign X and y
  y = df['y_val']
  X = df.loc[:, df.columns != 'y_val']
  X = X.drop(columns=['Date'])

  # Train / Test Split
  y_train, y_test= np.split(y, [date_index])
  X_train, X_test= np.split(X, [date_index])

  df.set_index('Date', inplace=True)

  print("X_train Shape: {}".format(X_train.shape))
  print("X_test Shape: {}".format(X_test.shape))

  return X_train, y_train, X_test, y_test

def train_val_split_ts(X_train, y_train, val_size):
  # Get Index
  date_index=int(X_train.shape[0]*(1-val_size))

  # Train / Test Split
  y_train, y_val= np.split(y_train, [date_index])
  X_train, X_val= np.split(X_train, [date_index])

  print("X_train Shape: {}".format(X_train.shape))
  print("X_val Shape: {}".format(X_val.shape))

  return X_train, y_train, X_val, y_val

# Fit LGBM Model
def fit_lightgbm (params, X_train, y_train, X_test, y_test):
  
  # Define Datasets
  lgb_train=lgb.Dataset(X_train, y_train)
  lgb_eval=lgb.Dataset(X_test, y_test, reference=lgb_train)
  
  # Train Model
  model = lgb.train(params,
                  train_set=lgb_train,
                  valid_sets=lgb_eval,
                  keep_training_booster=True,
                  early_stopping_rounds=30)
  
  return model

# Plot Results
def show_results(model, X_test, y_test):
  results = pd.DataFrame([y_test.reset_index(drop=True), model.predict(X_test)])
  results = results.transpose()
  results.columns=['Actual', 'Predicted']
  # print(results)

  fig, ax = plt.subplots(figsize=(15, 5))
  results["Actual"].plot(ax=ax, label='Actual Values', title='Prediction vs. Actual Values')
  results["Predicted"].plot(ax=ax, label='Predicted Values')
  ax.legend(['Actual Values', 'Predicted Values'])
  plt.show()

# Hyperparameter Tuning - Bayesian Optimization
def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=3, random_seed=1945,n_estimators=1000, param_bounds=param_bounds, output_process=False):
  # Prepare Data
  train_data = lgb.Dataset(X, y, free_raw_data=False)
  # Black Box Function
  def lgb_eval(learning_rate,num_leaves, feature_fraction, bagging_fraction, max_depth, max_bin, min_data_in_leaf,min_sum_hessian_in_leaf,subsample):
    # Ensure Parameters are correct types, valid values
    params = {'objective':'regression', 'metric':'rmse'}
    params['learning_rate'] = max(min(learning_rate, 1), 0)
    params["num_leaves"] = int(round(num_leaves))
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = int(round(max_depth))
    params['max_bin'] = int(round(max_depth))
    params['min_data_in_leaf'] = int(round(min_data_in_leaf))
    params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
    params['subsample'] = max(min(subsample, 1), 0)
    # Black Box Function
    cv_result = lgb.cv(params = params, 
                    train_set=train_data, 
                    folds=tss, 
                    seed=1945, 
                    stratified=False, #True for Classification, False for Regression
                    verbose_eval =200,
                    early_stopping_rounds=10, 
                    metrics=['rmse'])
    # Return value to maximize (minimize MSE --> maximize negative MSE)
    return -1 * cv_result['rmse-mean'][-1]

  # Bayesian Optimizer
  optimizer = BayesianOptimization(
      f = lgb_eval,
      pbounds = param_bounds,
      random_state = random_seed
  )
  
  # Maximize Optimizer
  optimizer.maximize(
    init_points=init_round,
    n_iter=opt_round
    )
  
  # Track Model Results
  model_results = []
  for model in range(len(optimizer.res)):
    model_results.append(optimizer.res[model]['target'])

  # Return Optimal Parameter Set
  return optimizer.res[pd.Series(model_results).idxmax()]['target'], optimizer.res[pd.Series(model_results).idxmax()]['params']

