import numpy as np

def select_class_subset(X_train, X_test, y_train, y_test, ratio=ratio):

    unique_classes = np.unique(y_train)
    num_classes_to_select = int(len(unique_classes) * ratio)

    selected_classes = np.random.choice(unique_classes, num_classes_to_select, replace=False)

    train_mask = np.isin(y_train, selected_classes)
    X_train_filtered, y_train_filtered = X_train[train_mask], y_train[train_mask]

    test_mask = np.isin(y_test, selected_classes)
    X_test_filtered, y_test_filtered = X_test[test_mask], y_test[test_mask]

    return X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered, selected_classes
