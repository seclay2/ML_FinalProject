def SGD(self, X_train_scaled, X_test_scaled, y_train, y_test, nIteration=200):

    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_squared_error

    clf = SGDRegressor(max_iter=nIteration, tol=1e-3)
    clf.fit(X_train_scaled, y_train.values.ravel())
    y_pred = clf.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred)
    return y_pred, rmse
