def Q_05(self, full_dataset, columns_to_keep):
    # Task 5: Given a full_dataset (as Pandas dataframe) having all the 37 columns and certain number of rows
    # and columns_to_keep (as Pandas Dataframe) having 2 columns {variable_name,corr_score_with_target} similar
    # to the one you computed in Q_04.
    # Please return a reduced full_dataset keeping only the columns listed in the columns_to_keep dataframe.
    #
    import pandas as pd



    #### YOUR CODE HERE####

    reduced_full_dataset = full_dataset

    # Keep only the column names
    reduced_full_dataset = reduced_full_dataset[columns_to_keep['variable_name']]

    # Rejoin BWEIGHT (as we later split into X and Y)
    reduced_full_dataset = reduced_full_dataset.join(full_dataset['BWEIGHT'])

    return reduced_full_dataset


