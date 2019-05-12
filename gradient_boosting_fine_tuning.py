
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


"""
1)

Tuning the number of boosting rounds:

Let's start with parameter tuning by seeing how the number
of boosting rounds (number of trees you build) impacts the out-of-sample performance of your XGBoost model.

working with the Ames housing dataset.

"""


# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective": "reg:linear", "max_depth": 3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse",
                        as_pandas=True, seed=123)

    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses, columns=["num_boosting_rounds", "rmse"]))


"""
OUTPUT:
   num_boosting_rounds          rmse
0                    5  50903.299479
1                   10  34774.194010
2                   15  32895.098958
"""

####################


"""
2)

Automated Boosting round selection using Early Stopping:

Now, instead of attempting to cherry pick the best possible number of boosting rounds, you can very easily
have XGBoost automatically select the number of boosting rounds for you within xgb.cv(). This is done using a technique
called early stopping.

Early stopping works by testing the XGBoost model after every boosting round against a hold-out dataset 
and stopping the creation of additional boosting rounds (thereby finishing training of the model early) 
if the hold-out metric ("rmse" in our case) does not improve for a given number of rounds.
Here you will use the early_stopping_rounds parameter in xgb.cv() with a large possible number of boosting rounds (50). 
Bear in mind that if the holdout metric continuously improves up through when num_boosting_rounds is reached, 
then early stopping does not occur.

"""

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix,folds=3,early_stopping_rounds=10,num_boost_round=50,params=params,metrics="rmse")

# Print cv_results
print(cv_results)

"""
OUTPUT:
    train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
0     139216.236979    11003.460718   136073.533854   20847.453786
1     100916.588541     8360.940664   100040.671875   20400.645712
2      74266.667969     6456.598227    75446.850260   20241.439411
3      55884.333333     5242.444592    58972.595052   19278.263295
4      43212.674479     4511.794575    48573.604166   18256.123115
5      34647.425781     3955.233378    42046.942057   17440.336903
6      28930.772787     3505.470795    38143.645182   16171.500949
7      25190.904297     3104.597775    35869.436849   15146.611817
8      22494.291667     2783.233912    34498.792318   14115.094395
9      20689.128255     2550.467530    33915.322916   13348.837037
10     19572.835286     2380.505677    33390.004557   12996.970336
11     18622.080078     2333.532159    33109.637370   12489.962565
12     17980.963542     2218.174235    32935.100911   12244.117756
13     17546.319987     2125.087809    32928.362631   12039.735171
14     17099.578451     2089.417219    32758.376302   11784.010967
15     16730.943359     2111.639547    32698.080729   11668.063282
16     16463.676758     2132.812095    32683.291666   11626.876512
17     16269.478190     2126.161486    32593.000000   11474.610217
18     15975.022787     2173.858455    32471.083985   11284.396350
19     15712.653320     2144.501656    32383.916016   11217.820410
20     15426.445313     2133.948225    32283.158854   10990.113432
21     15215.499674     2128.225268    32212.466146   10993.932510
22     15042.100586     2066.260517    32135.411458   11068.126755
23     14798.883789     2063.607809    32105.410807   11051.169790
24     14578.954753     2014.224408    32032.337891   10888.501998
25     14418.299805     1974.254228    32006.795573   10956.198724
26     14179.743164     1912.758671    31948.430339   11001.611446
27     13964.085287     1973.755878    31924.837239   10981.289079
28     13824.422852     1942.204236    31928.699219   11007.709662
29     13663.833985     2009.266768    31885.687500   11023.458088
30     13515.024414     1974.453169    31891.387370   11078.143713
31     13404.504883     1945.713066    31889.734375   11063.702025
32     13249.442708     1984.520151    31927.427084   11069.197031
33     13081.763997     1939.001541    31882.128255   10983.070760
34     12928.792643     1900.396151    31842.106120   11025.148597
35     12830.354167     1930.068975    31786.822916   10991.857811
36     12722.168620     1932.428311    31790.183594   11044.894643
37     12604.854818     1920.696769    31806.370443   11038.509213
38     12420.081055     1844.526245    31772.946614   11089.692542
39     12258.669271     1784.554713    31713.787760   11045.535600
40     12135.880208     1795.535175    31723.000000   11072.201488
41     11963.071940     1834.066326    31649.097005   11062.160411
42     11823.346029     1866.911338    31626.211589   11077.289119
43     11677.214844     1940.146153    31644.631510   11074.537585
44     11558.674805     1958.367760    31692.708984   11103.414558
45     11390.392903     1951.893977    31688.261067   11059.248096
46     11242.788086     1924.091801    31676.481120   11042.301313
47     11104.977864     1891.535074    31654.587240   11024.065677
48     10967.748372     1889.090649    31691.982422   11065.005784
49     10887.622070     1870.142060    31654.694661   11097.172233
"""


####################

"""
3)

Tuning eta / Learning Rate: 


The learning rate in XGBoost is a parameter that can range between 0 and 1, 
with higher values of "eta" penalizing feature weights more strongly, causing much stronger regularization.

"""

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective": "reg:linear", "max_depth": 3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta
for curr_val in eta_vals:
    params["eta"] = curr_val

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, folds=3, metrics="rmse", seed=123, num_boost_round=10,
                        early_stopping_rounds=5)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta", "best_rmse"]))

"""
OUTPUT:
     eta      best_rmse
0  0.001  184300.723958
1  0.010  169550.255208
2  0.100   75405.712240

"""


"""
4)

Tuning max_depth:

max_depth, which is the parameter that dictates the maximum depth that each tree in a boosting round can grow to.
Smaller values will lead to shallower trees, and larger values to deeper trees.

"""

# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary
params = {"objective": "reg:linear"}

# Create list of max_depth values
max_depths = [2, 5, 10, 20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:
    params["max_depth"] = curr_val

    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, folds=2, num_boost_round=10, early_stopping_rounds=5,
                        metrics="rmse", seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)), columns=["max_depth", "best_rmse"]))

"""
   max_depth     best_rmse
0          2  36132.386068
1          5  32000.271484
2         10  32862.851563
3         20  32805.277995
"""

####################

"""
5)

Tuning colsample_bytree:

You've already seen this if you've ever worked with scikit-learn's RandomForestClassifier or RandomForestRegressor,
where it just was called max_features. In both xgboost and sklearn, 
this parameter (although named differently) simply 
specifies the fraction of features to choose from at every split in a given tree.
In xgboost, colsample_bytree must be specified as a float between 0 and 1.


There are several other individual parameters that you can tune, such as "subsample", 
which dictates the fraction of the training data that is used during any given boosting round.

fraction is low : then underfitting
high : Overfitting

"""
# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary
params = {"objective": "reg:linear", "max_depth": 3}

# Create list of hyperparameter values: colsample_bytree_vals
colsample_bytree_vals = [0.1, 0.5, 0.8, 1]
best_rmse = []

# Systematically vary the hyperparameter value
for curr_val in colsample_bytree_vals:
    params["colsample_bytree"] = curr_val

    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                        num_boost_round=10, early_stopping_rounds=5,
                        metrics="rmse", as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree", "best_rmse"]))


"""
       colsample_bytree     best_rmse
    0               0.1  45017.404296
    1               0.5  36050.654297
    2               0.8  35372.572266
    3               1.0  35836.046875
    

"""


"""
6) Grid Search CV:

To find the best model exhaustively from a collection of possible parameter values across multiple parameters simultaneously.

"""


# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm,param_grid=gbm_param_grid,cv=4,scoring="neg_mean_squared_error",verbose=1)


# Fit grid_mse to the data
grid_mse.fit(X,y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

"""
   OUTPUT:
    Fitting 4 folds for each of 4 candidates, totalling 16 fits
    Best parameters found:  {'colsample_bytree': 0.7, 'max_depth': 5, 'n_estimators': 50}
    Lowest RMSE found:  30342.16964561695

"""

########

"""
7) RandomSearch CV:

Just a subset of parameter combinations are taken into consideration. 


"""


# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': np.arange(2, 12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators=10)

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(estimator=gbm,param_distributions=gbm_param_grid,scoring="neg_mean_squared_error",n_iter=5,cv=4)


# Fit randomized_mse to the data
randomized_mse.fit(X,y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

"""
    Best parameters found:  {'n_estimators': 25, 'max_depth': 5}
    Lowest RMSE found:  36636.35808132903

"""