def Q_14(self, batch_GD_result, stochastic_GD_result, minibatch_GD_result):
    # Task 14: Given the 3 sets of tuples from the 3 experiments with batch gradient descent,
    # stochastic gradient descent and mini-batch gradient descent, return a string from the set
    # {"batch-GD", "stochastic-GD", "minibatch-GD"} that demonstrated the best predictive performance
    # in terms of RMSE.

    (beta_B, y_pred_B, RMSE_B, cpu_time_B) = batch_GD_result
    (beta_S, y_pred_S, RMSE_S, cpu_time_S) = stochastic_GD_result
    (beta_M, y_pred_M, RMSE_M, cpu_time_M) = minibatch_GD_result

    ## YOUR CODE HERE ##
    best_rmse = "TO-BE-DETERMINED"

    # Calculate which RMSE value is the lowest

    if RMSE_B < RMSE_S and RMSE_B < RMSE_M:
        best_rmse = "batch-GD"

    if RMSE_S < RMSE_B and RMSE_S < RMSE_M:
        best_rmse = "stochastic-GD"

    if RMSE_M < RMSE_B and RMSE_M < RMSE_S:
        best_rmse = "minibatch-GD"

    return best_rmse