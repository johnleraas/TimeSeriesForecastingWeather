import pandas as pd

#  For LGBM model, get feature importance
def get_feature_importance(model):
  df = pd.DataFrame({'Feature':model.feature_name(), 'Value':model.feature_importance()})
  df = df.sort_values(by="Value", ascending=False)
  df["Value_pct"] = df["Value"]/(df["Value"].sum())
  
  return df

def plot_importance (df, num, title, feat_col="Feature", count_col="Value_pct", fig_size = (20,15)):
  df_plt = df.sort_values(by=count_col, ascending=True)
  plt.figure(figsize=fig_size)
  plt.barh(df_plt[feat_col][df.shape[0]-num:df.shape[0]], df_plt[count_col][df.shape[0]-num:df.shape[0]])
  plt.ylabel('Feature')
  plt.xlabel('Importance')
  plt.title('Feature Importance')
  
  
def agg_importance(df_agg):
  agg_col = 'agg'
  df_agg[agg_col] = 0

  #y_val_lag
  prefix = 'y_val_lag'
  for i in range(0, df_agg.shape[0]):
    if df_agg.loc[i, 'Feature'].startswith(prefix):
      df_agg.loc[i, agg_col] = prefix

  #y_val_lag
  prefix = 'y_val_rollwind'
  for i in range(0, df_agg.shape[0]):
    if df_agg.loc[i, 'Feature'].startswith(prefix):
      df_agg.loc[i, agg_col] = prefix

  #fed_holiday_lag
  prefix = 'fed_holiday_lag'
  for i in range(0, df_agg.shape[0]):
    if df_agg.loc[i, 'Feature'].startswith(prefix):
      df_agg.loc[i, agg_col] = prefix

  # Holiday
  hol_list = ["NY", "MLK", "PRES", "MEM", "JUN", "JUL4", "LAB", "COLUMB", "VET", "TGD", "XMAS"]
  for i in range(0, df_agg.shape[0]):
    if hol_list.count(df_agg.loc[i, 'Feature']) > 0:
      df_agg.loc[i, agg_col] = "Holiday"

  # Temperature
  temp_list = ["tmin", "tmax", "tavg"]
  for i in range(0, df_agg.shape[0]):
    if temp_list.count(df_agg.loc[i, 'Feature']) > 0:
      df_agg.loc[i, agg_col] = "Temperature"

  # Fill in 0s
  for i in range(0, df_agg.shape[0]):
    if df_agg.loc[i, agg_col]==0:
      df_agg.loc[i, agg_col] = df_agg.loc[i, 'Feature']


  # Aggregate
  df_agg = df_agg[['agg', 'Value_pct']].groupby('agg').sum()

  # Sort Values
  df_agg = df_agg.sort_values("Value_pct", ascending=False)

  # Reset Index
  df_agg.reset_index(inplace=True)
  
  return df_agg
