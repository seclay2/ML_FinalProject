# Test driver

import numpy as np
from Tasks import *

q = Tasks(full_dataset_filename='./dataset/Training_Data.csv',
             judge_filename='dataset/Training_Data_9col.csv')

one_hot_dataset = q.Q_02(q.judge_dataset)
q.EDA(one_hot_dataset)

# np.set_printoptions(threshold=np.inf)
# y_pred = q.Q_17(q.judge_dataset)
# print("y pred: \n", y_pred)




# --- KEV MAIN ---
# print("\n\nQuestion 1")
# print(q.Q_01())
#
#
# print("\n\nQuestion 2")
# one_hot_dataset = q.Q_02(q.full_dataset)
# print(one_hot_dataset)
#
#
# print("\n\nQuestion 3")
# missing_count, revised_full_dataset = q.Q_03(one_hot_dataset)
# print(missing_count)
#
#
# print("\n\nQuestion 4")
# corr_dataset = q.Q_04(one_hot_dataset)
# print(corr_dataset)
#
# print("\n\nQuestion 5")
# reduced_full_dataset = q.Q_05(revised_full_dataset, corr_dataset)
# print(reduced_full_dataset)
#
#
# print("\n\nQuestion 6")
# # X, y = q.Q_06(revised_full_dataset)
# X, y = q.Q_06(reduced_full_dataset)
#
# print(X)
# print(y)
#
# print("\n\nQuestion 7")
# X_train, X_test, y_train, y_test = q.Q_07(X, y)
# print(X_train)
#
# print("\n\nQuestion 8")
# X_train, X_test, y_train, y_test = q.Q_08(X_train, X_test, y_train, y_test)
# print(X_train)
#
# print("\n\nQuestion 9")
# # X_train, X_test, y_train, y_test = q.Q_09(X_train, X_test, y_train, y_test)
# # print(X_train)
#
# print("\n\nQuestion 10")
# beta, y_pred, RMSE = q.Q_10(X_train, X_test, y_train, y_test)
# print(beta)
# print(y_pred)
# print(RMSE)
#
# print("\n\nQuestion 11")
# batch = q.Q_11(X_train, X_test, y_train, y_test)
# beta, y_pred, RMSE, cpu_time = batch
# print(beta)
# print(y_pred)
# print(RMSE)
# print(cpu_time)
#
# print("\n\nQuestion 12")
# stochastic = q.Q_12(X_train, X_test, y_train, y_test)
# beta, y_pred, RMSE, cpu_time = stochastic
# print(beta)
# print(y_pred)
# print(RMSE)
# print(cpu_time)
#
# print("\n\nQuestion 13")
# minibatch = q.Q_13(X_train, X_test, y_train, y_test)
# beta, y_pred, RMSE, cpu_time = minibatch
# print(beta)
# print(y_pred)
# print(RMSE)
# print(cpu_time)
#
# print("\n\nQuestion 14")
# best_rmse = q.Q_14(batch, stochastic, minibatch)
# print(best_rmse)
#
# print("\n\nQuestion 15")
# best_time = q.Q_15(batch, stochastic, minibatch)
# print(best_time)
#
# print("\n\nQuestion 16")
# summary_16 = q.Q_16(X_train, X_test, y_train, y_test)
# learning_rate, beta, summary = summary_16
# print(learning_rate)
# print(beta)
# print(summary)
#
# print("\n\nQuestion 17")
# y_predictions = q.Q_17(q.judge_dataset)
# print(y_predictions)