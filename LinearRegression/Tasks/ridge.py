def ridge(self, X_train_scaled, X_test_scaled, y_train, y_test, alpha=0.1):

    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    clf = Ridge(alpha=alpha)
    clf.fit(X_train_scaled, y_train.values.ravel())
    y_pred = clf.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred)
    return y_pred, rmse
