def linear_regression(self, X_train_scaled, X_test_scaled, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    reg = LinearRegression().fit(X_train_scaled, y_train)
    y_pred = reg.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred)

    return y_pred, rmse
