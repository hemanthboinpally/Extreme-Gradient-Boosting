#Linear Regression
import xgboost as xg
from sklearn.model _selection import train_test_split


"""Using Decision Trees as Base Learners : 

By default, XGBoost uses trees as base learners, so you don't have to specify that you want to use trees here with booster="gbtree".
xgboost has been imported as xgb and the arrays for the features and the target are available in X and y, respectively.

"""


# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.2, random_state=123)

# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(seed=123, objective="reg:linear",n_estimators=10 )

# Fit the regressor to the training set
fit = xg_reg.fit(X_train,y_train)

# Predict the labels of the test set: preds
preds = fit.predict(X_test)

# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

#Linear base Learners

"""
Linear Base Learners : 

This model, although not as commonly used in XGBoost, allows you to create a regularized linear regression using XGBoost's powerful learning API.
Create a param dictionary like that is feed in the cross validation function.  

The key-value pair that defines the booster type (base model) you need is "booster":"gblinear".

Create the DMatrix objects required by the XGBoost learning API.

"""

# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(data=X_train , label=y_train)
DM_test =  xgb.DMatrix(data=X_test , label=y_test)

# Create the parameter dictionary: params
params = {"booster":"gblinear", "objective":"reg:linear"}

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)

# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))


#Model Evaluation:

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))