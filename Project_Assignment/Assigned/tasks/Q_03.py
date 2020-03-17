def Q_03(self, full_dataset):
    # Task 3: Given the full_dataset (Pandas Dataframe), check if there are missing values, and if yes,
    # count how many, and impute the missing values with corresponding mean values.
    # Finally, return the counting result as a Pandas dataframe with 2 columns
    #  {variable_name,num_of_missing_values).  Please make sure the result lists all the variables
    #  (including the target) in the given dataset. Also, return the revised full_dataset after the missing
    # value imputations is done. Return these two pandas dataframe as tuple.
    import pandas as pd

    missing_count = pd.DataFrame()
    revised_full_dataset = pd.DataFrame()

    ## YOUR CODE HERE ##

    # Get the number of missing values
    missing_count = full_dataset.isnull().sum().to_frame().reset_index()

    # Create column headings
    missing_count.columns = ['variable_name', 'num_of_missing_values']

    # Replace the missing values with the mean
    revised_full_dataset = full_dataset.fillna(full_dataset.mean())


    return (missing_count, revised_full_dataset)
