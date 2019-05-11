#Linear Regression
import xgboost as xg
from sklearn.model _selection import train_test_split

#Using Decision Trees as Base Learners
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