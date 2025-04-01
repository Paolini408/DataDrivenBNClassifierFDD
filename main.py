import pandas as pd
from utils.feature_selections import features_selection_continuous, features_selection_discrete_entropy
from utils.bayesian_classifiers import cgn_classifier, kde_classifier, tan_classifier
from utils.performance_metrics import print_performance_and_compute_precision_and_recall, plot_precision_and_recall

if __name__ == "__main__":
    path = './data/summer_data_preprocessed.csv'
    df = pd.read_csv(path)

    sorted_labels = sorted(df['label'].unique())
    print(sorted_labels)

    n_input_features = 10  # Number of input features to be selected (based on availability and complexity)

    ### CGN ###
    alpha = 0.99  # Lower alpha means lower False Alarm Rate (0 < alpha < 1)
    X_columns_cgn = features_selection_continuous(df, n_input_features)
    lista_col_cgn = X_columns_cgn.columns.tolist()
    y_test, y_pred = cgn_classifier(df, lista_col_cgn, sorted_labels, alpha)
    precision, recall = print_performance_and_compute_precision_and_recall(y_test, y_pred, sorted_labels, 'CGN')
    plot_precision_and_recall(precision, recall, sorted_labels, 'CGN')

    ### KDE ###
    Lp = 2.5 * (10 ** -3)  # Lower Lp means lower False Alarm Rate (0 < Lp < +inf)
    X_columns_kde = features_selection_continuous(df, n_input_features)
    lista_col_kde = X_columns_kde.columns.tolist()
    y_test, y_pred = kde_classifier(df, lista_col_kde, sorted_labels, Lp)
    precision, recall = print_performance_and_compute_precision_and_recall(y_test, y_pred, sorted_labels, 'KDE')
    plot_precision_and_recall(precision, recall, sorted_labels, 'KDE')

    ### TAN ###
    max_bins = 5  # Maximum number of bins for discretization
    cost_ratio_list = [1] * (len(sorted_labels) - 1)  # In this case, all Fault costs are the same
    cost_ratio_list.append(1)  # Normal cost is the last one and can (should) be higher than the others to reduce FAR
    X_columns_tan = features_selection_discrete_entropy(df, max_bins, n_input_features)
    lista_col_tan = X_columns_tan.columns.tolist()
    y_test, y_pred = tan_classifier(X_columns_tan, df['label'], lista_col_tan, sorted_labels, cost_ratio_list)
    precision, recall = print_performance_and_compute_precision_and_recall(y_test, y_pred, sorted_labels, 'TAN')
    plot_precision_and_recall(precision, recall, sorted_labels, 'TAN')
