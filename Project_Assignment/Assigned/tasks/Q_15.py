def Q_15(self, batch_GD_result, stochastic_GD_result, minibatch_GD_result):
    # Task 15: Given the 3 sets of tuples from the 3 experiments with batch gradient descent,
    # stochastic gradient descent and mini-batch gradient descent, return a string from the set
    # {"batch-GD", "stochastic-GD", "minibatch-GD"} that demonstrated the least training time.

    (beta_B, y_pred_B, RMSE_B, cpu_time_B) = batch_GD_result
    (beta_S, y_pred_S, RMSE_S, cpu_time_S) = stochastic_GD_result
    (beta_M, y_pred_M, RMSE_M, cpu_time_M) = minibatch_GD_result

    ## YOUR CODE HERE ##
    best_time = "TO-BE-DETERMINED"

    # Calculate which cpu time is the lowest
    if cpu_time_B < cpu_time_S and cpu_time_B < cpu_time_M:
        best_time = "batch-GD"

    if cpu_time_S < cpu_time_B and cpu_time_S < cpu_time_M:
        best_time = "stochastic-GD"

    if cpu_time_M < cpu_time_B and cpu_time_M < cpu_time_S:
        best_time = "minibatch-GD"

    return best_time