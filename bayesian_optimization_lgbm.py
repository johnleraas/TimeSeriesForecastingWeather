from bayes_opt import BayesianOptimization
from scipy.ndimage.measurements import mean
import lightgbm as lgb
# 
# Hyperparameter Tuning - Bayesian Optimization
def bayes_parameter_opt_lgb(X, y, param_bounds, tss, init_round=15, opt_round=25, n_folds=3, random_seed=1945,n_estimators=1000, output_process=False):
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
