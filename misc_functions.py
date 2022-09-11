# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 08:25:18 2022

"""
from datetime import datetime
import pandas as pd
import numpy as np
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
