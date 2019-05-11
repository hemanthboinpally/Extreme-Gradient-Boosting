

# Import xgboost
import xgboost as xgb


"""
XGBoost: Fit/Predict

Churn Analysis of the data:


This dataset contains imaginary data from a ride-sharing app with user behaviors over their first month of app usage in a set of imaginary cities
as well as whether they used the service 5 months after sign-up. It has been pre-loaded for you into a DataFrame called churn_data!


Index(['avg_dist', 'avg_rating_by_driver', 'avg_rating_of_driver', 'avg_inc_price', 'inc_pct', 'weekday_pct', 'fancy_car_user', 'city_Carthag', 'city_Harko', 'phone_iPhone',
       'first_month_cat_more_1_trip', 'first_month_cat_no_trips', 'month_5_still_here'],
      dtype='object')

avg_dist  avg_rating_by_driver  avg_rating_of_driver  avg_inc_price  inc_pct  ...  city_Harko  phone_iPhone  first_month_cat_more_1_trip  first_month_cat_no_trips  month_5_still_here
0      3.67                   5.0                   4.7           1.10     15.4  ...           1             1                            1                         0                   1
1      8.26                   5.0                   5.0           1.00      0.0  ...           0             0                            0                         1                   0
2      0.77                   5.0                   4.3           1.00      0.0  ...           0             1                            1                         0                   0
3      2.36                   4.9                   4.6           1.14     20.0  ...           1             1                            1                         0                   1
4      3.13                   4.9                   4.4           1.19     11.8  ...           0             0                            1                         0                   0

Goal is to use the first month's worth of data to predict
whether the app's users will remain users of the service at the 5 month mark.
This is a typical setup for a churn prediction problem.


"""


# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


"""

Measuring the Accuracy: 

XGBoost's learning API:

XGBoost gets its lauded performance and efficiency gains by utilizing its own optimized data structure for datasets called a DMatrix.

In the above model the input datasets were converted into DMatrix data on the fly, but when you use the xgboost cv object, 
you have to first explicitly convert your data into a DMatrix

reg:logistic gives the probability of belonging to a particular class.

params is your parameter dictionary, 
folds is the number of cross-validation folds (3),
num_boosting_rounds is the number of trees we want to build (5), 
metrics is the metric you want to compute (this will be "error")

"""



# Create the DMatrix: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=churn_data.iloc[ :,: -1], label=churn_data.month_5_still_here)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))


"""
Metrics : AUC Measure
"""

# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])