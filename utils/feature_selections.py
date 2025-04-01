import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif


### CONTINUOUS DISCRETIZATION ###
def features_selection_continuous(df, n_features):
    X = df.loc[:, df.columns != 'label']
    # Remove categorical features (states) and redundant features (use heating and cooling coil power instead)
    X = X.drop(['HUM_state', 'CC_state', 'PostHC_state', 'T_F_out_CC', 'T_F_in_CC', 'T_F_out_PostHC', 'T_F_in_PostHC',
                'V_F_in_CC', 'V_F_in_PostHC'], axis=1)
    Y = df['label']
    n_bins = 100000  # High number of bins to consider continuous features
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_bins = discretizer.fit_transform(X)
    X_discrete = pd.DataFrame(X_bins, columns=X.columns)
    mi_scores = mutual_info_classif(X_discrete, Y)
    mi_scores_df = pd.DataFrame({'Feature': X_discrete.columns, 'MI Score': mi_scores})
    mi_scores_sorted = mi_scores_df.sort_values(by='MI Score', ascending=False).reset_index(drop=True)
    selected_features = mi_scores_sorted.iloc[:n_features]['Feature'].tolist()
    X_selected_cont = X[selected_features]
    return X_selected_cont


### EQUAL WIDTH DISCRETIZATION ###
def features_selection_discrete(df, n_bins, n_features):
    X = df.loc[:, df.columns != 'label']
    # Remove categorical features (states) and redundant features (use heating and cooling coil power instead)
    X = X.drop(['HUM_state', 'CC_state', 'PostHC_state', 'T_F_out_CC', 'T_F_in_CC', 'T_F_out_PostHC', 'T_F_in_PostHC',
                'V_F_in_CC', 'V_F_in_PostHC'], axis=1)
    Y = df['label']
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_bins = discretizer.fit_transform(X)
    X_discrete = pd.DataFrame(X_bins, columns=X.columns)
    mi_scores = mutual_info_classif(X_discrete, Y)
    mi_scores_df = pd.DataFrame({'Feature': X_discrete.columns, 'MI Score': mi_scores})
    mi_scores_sorted = mi_scores_df.sort_values(by='MI Score', ascending=False).reset_index(drop=True)
    selected_features = mi_scores_sorted.iloc[:n_features]['Feature'].tolist()
    X_selected_disc = X_discrete[selected_features]
    return X_selected_disc


### ENTROPY DISCRETIZATION ###
def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return entropy(probabilities, base=2)


def weighted_entropy_multiple(x_train, y_train, cut_points):
    bins = [-np.inf] + sorted(cut_points) + [np.inf]
    bin_indices = np.digitize(x_train, bins) - 1  # Output: bin membership list
    weighted_entropy = 0
    for bin_index in range(len(bins) - 1):
        bin_mask = bin_indices == bin_index  # Output: array booleano (True False ... False)
        bin_entropy = calculate_entropy(y_train[bin_mask])
        weighted_entropy += (np.sum(bin_mask) * bin_entropy) / len(y_train)
    return weighted_entropy


def find_best_splits(feature_data, y_train, n_bins_max):
    # Sorted column of x_train['feature'] maintaining the correspondence with y_train
    combined = sorted(zip(feature_data, y_train), key=lambda x: x[0])
    reduced_combined = combined[::10]  # Select every k-th element (k=1,10,100)
    # Unpack the reduced combined pairs back into separate lists
    reduced_feature_data, reduced_y_train = zip(*reduced_combined)
    reduced_feature_data = np.array(reduced_feature_data)
    reduced_y_train = np.array(reduced_y_train)
    unique_values = np.unique(reduced_feature_data)  # Only unique values (different from each other)
    selected_cut_points = []
    while len(selected_cut_points) < (n_bins_max - 1):  # bins = cut points + 1
        best_cut_point = None
        min_entropy = float('inf')  # Evaluate min_entropy < inf, otherwise n_bins will always be equal to n_bins_max
        for i in range(len(unique_values) - 1):
            cut_point = (unique_values[i] + unique_values[i + 1]) / 2
            if cut_point in selected_cut_points:
                continue
            current_entropy = weighted_entropy_multiple(reduced_feature_data, reduced_y_train,
                                                        selected_cut_points + [cut_point])
            if current_entropy < min_entropy:
                min_entropy = current_entropy
                best_cut_point = cut_point
        if best_cut_point is not None:
            selected_cut_points.append(best_cut_point)
        else:
            break
    return [-np.inf] + sorted(selected_cut_points) + [np.inf]


def obtain_X_entropy_disc(x_train, y_train, n_bins_max):
    best_splits = {}
    X_entropy = pd.DataFrame()
    for col in x_train.columns:
        best_splits[col] = find_best_splits(x_train[col], y_train, n_bins_max)
        X_entropy[col] = np.digitize(x_train[col], best_splits[col], right=False)
    return X_entropy


def features_selection_discrete_entropy(df, n_bins_max, n_features):
    X = df.loc[:, df.columns != 'label']
    # Remove categorical features (states) and redundant features (use heating and cooling coil power instead)
    X = X.drop(['HUM_state', 'CC_state', 'PostHC_state', 'T_F_out_CC', 'T_F_in_CC', 'T_F_out_PostHC', 'T_F_in_PostHC',
                'V_F_in_CC', 'V_F_in_PostHC'], axis=1)
    Y = df['label']
    X_discrete = obtain_X_entropy_disc(X, Y, n_bins_max)
    mi_scores = mutual_info_classif(X_discrete, Y)
    mi_scores_df = pd.DataFrame({'Feature': X_discrete.columns, 'MI Score': mi_scores})
    mi_scores_sorted = mi_scores_df.sort_values(by='MI Score', ascending=False).reset_index(drop=True)
    selected_features = mi_scores_sorted.iloc[:n_features]['Feature'].tolist()
    X_selected_disc = X_discrete[selected_features]
    return X_selected_disc
