# feature_target_split
#
# Separates the full_dataset into two parts: X and y, where X denotes the input matrix
# containing only the input variables, and y denotes the target vector containing only the target values


def feature_target_split(self, full_dataset, target_str):
    # Input without target
    X = full_dataset.drop([target_str], axis=1)

    # Target
    y = full_dataset[[target_str]]

    return X, y
