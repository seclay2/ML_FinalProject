def Q_09(self, X_train, X_test, y_train, y_test):
    # Task 9: Given the 4 splits denoting the training and test dataset,
    # Apply standardization (i.e., normalization) scaling on the training dataset (X_train).
    # Then scale the test dataset based on the metrics you obtain when you scale the training dataset.
    # PLEASE DO NOT SCALE y_train and y_test.
    # Finally, return as a tuple the scaled X_train, X_test and the intact y_train and y_test.
    import pandas as pd
    X_train_scaled = pd.DataFrame()
    X_test_scaled = pd.DataFrame()


    ### YOUR CODE HERE ###

    # Assuming standard deviation is never 0
    X_train_scaled = (X_train - X_train.mean()) / X_train.std()

    X_test_scaled = (X_test - X_train.mean()) / X_train.std()

    return (X_train_scaled, X_test_scaled, y_train, y_test)

