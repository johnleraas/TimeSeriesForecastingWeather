import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Plot raw data
def plot_rawdata(df):
  sns.set_style("white")
  plt.figure(figsize=(15,5))
  sns.scatterplot(x=df.index, y=df.y_val, data=df)
  plt.xlabel('Date', fontsize = 20)
  plt.ylabel('Gross Sales', fontsize = 20)
  plt.ylim(0, )
  plt.title('Gross Sales vs. Time', fontsize = 26)
  plt.tight_layout()

  plt.show()
  
#Plot Train / Test / Val Split
def plot_trainValTest_split(df, y_train, y_val, y_test):
  # Get Indices
  train_start = 0
  train_end = y_train.shape[0]

  val_start=train_end
  val_end=val_start+y_val.shape[0]

  test_start=val_end
  test_end=test_start+y_test.shape[0]

  # Assign Datasets
  data_train = pd.DataFrame({'Date':pd.to_datetime(df.index[train_start:train_end]), 
                            'y_val': y_train})

  data_val = pd.DataFrame({'Date':pd.to_datetime(df.index[val_start:val_end]), 
                            'y_val': y_val})

  data_test = pd.DataFrame({'Date':pd.to_datetime(df.index[test_start:test_end]), 
                            'y_val': y_test})

  # Plot
  fig, ax = plt.subplots(figsize=(15, 5))
  plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
  data_train.plot(x='Date', y='y_val',ax=ax, label = 'Training Data')
  data_val.plot(x='Date', y='y_val', ax=ax, label= 'Validation Data')
  data_test.plot(x='Date', y='y_val', ax=ax, label= 'Test Data')
  plt.xlabel('Date', fontsize = 20)
  plt.ylabel('Gross Sales', fontsize = 20)
  fig.suptitle('Train / Validation / Test Split', fontsize=24)

  plt.show()
  
# Plot Kfold split
def plot_kfold_split(df, y_train, n_folds):
  df = pd.DataFrame({'Date': pd.to_datetime(df.index[0:y_train.shape[0]]),
                      'y_val': y_train}).set_index('Date')

  fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

  fold = 0
  for train_idx, val_idx in tss.split(df):
      train = df.iloc[train_idx]
      test = df.iloc[val_idx]
      train['y_val'].plot(ax=axs[fold],
                          label='Training Set',
                          title=f'Data Train/Test Split Fold {fold}')
      test['y_val'].plot(ax=axs[fold],
                          label='Test Set')
      axs[fold].axvline(test.index.min(), color='black', ls='--')
      fold += 1
  plt.show()


# Plot Importance
def plot_importance (df, num, title, feat_col="Feature", count_col="Value_pct", fig_size = (15,10)):
  df_plt = df.sort_values(by=count_col, ascending=True)
  plt.figure(figsize=fig_size)
  plt.barh(df_plt[feat_col][df.shape[0]-num:df.shape[0]], df_plt[count_col][df.shape[0]-num:df.shape[0]])
  plt.ylabel('Feature', fontsize=20)
  plt.xlabel('Importance', fontsize=20)
  plt.yticks(weight='bold')
  plt.title('Feature Importance', fontsize=26)
  plt.show()
  
# Plot Predictions
def plot_predictions(model, df, X_test, y_test, date_start, date_end, date_interval=5, title="Predicted vs. Actual Values"):

  results = pd.DataFrame({'Date': pd.to_datetime(df.index[date_start:date_end]),
                        'Actual': y_test.reset_index(drop=True),
                        'Predicted': model.predict(X_test)}).set_index('Date')

  fig, ax = plt.subplots(figsize=(15, 5), sharex=True)

  results["Actual"].plot(ax=ax, 
                         x=results.index,
                         label='Actual Values')
  results["Predicted"].plot(ax=ax, 
                            x=results.index,
                            label='Predicted Values')
  plt.ylabel('Gross Sales', fontsize=20)
  plt.xlabel('Date', fontsize=20)
  #plt.title('Feature Importance', fontsize=26)
  plt.suptitle(title, fontsize=24)
  ax.legend(['Actual Values', 'Predicted Values'])
  plt.ylim(0, )
  # ax.xaxis.set_major_locator(mdates.DayLocator(interval = date_interval))
  # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
  # plt.gcf().autofmt_xdate()

  plt.show()
