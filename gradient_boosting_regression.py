#Linear Regression

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

