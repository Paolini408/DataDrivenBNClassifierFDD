import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from scipy.stats import f
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mutual_info_score
import networkx as nx


### CONDITIONAL GAUSSIAN NETWORK (CGN) ###
def cgn_classifier(dataframe, var_list, lab_sorted, alpha_value):
    """
    Performs classification using a Conditional Gaussian Network (CGN) approach.

    Args:
    - dataframe (DataFrame): The pandas DataFrame containing the dataset.
    - var_list (list): A list of feature variables to be used for classification.
    - lab_sorted (list): A sorted list of unique class labels in the dataset.
    - alpha_value (float): The significance level for calculating the critical value.

    Returns:
    - y_test (list): True class labels for the test data.
    - y_pred (list): Predicted class labels for the test data.
    """
    n_features_CGN = len(var_list)
    X = dataframe.loc[:, var_list]
    Y = dataframe['label']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, stratify=Y)

    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    label_train_means = {}
    label_train_covariances = {}
    label_train_sizes = {}
    for class_label in lab_sorted:
        label_train_data = x_train[y_train == class_label]
        label_train_means[class_label] = label_train_data.mean()
        cov_matrix = label_train_data.cov()
        epsilon = 1E-9
        cov_matrix_reg = cov_matrix + np.eye(cov_matrix.shape[0]) * epsilon
        label_train_covariances[class_label] = cov_matrix_reg
        label_train_sizes[class_label] = label_train_data.shape[0]

    nc = len(lab_sorted)
    prior_probs = {}
    for class_label in lab_sorted:
        prior_p = (label_train_sizes[class_label] + 1) / (len(y_train) + nc)
        prior_probs[class_label] = prior_p
        # Prior probabilities can also be evaluated based on historical experience and domain expertise

    alpha = alpha_value  # Set in advance according to the user's tolerance in order to reduce FAR
    N = label_train_sizes['Normal']
    m = n_features_CGN
    F = f.ppf(1 - alpha, m, N - m)
    CL_value = (m * (N + 1) * (N - 1) * F) / (N * (N - m))

    y_pred = []
    Lp_list = []
    for _, row_test_data in x_test.iterrows():
        instance_probs = {}
        for class_label in label_train_means.keys():
            mean = label_train_means[class_label]  # Containing every feature of 'row_test_data'
            cov = label_train_covariances[class_label]
            instance_probs[class_label] = multivariate_normal.pdf(row_test_data, mean=mean, cov=cov,
                                                                  allow_singular=True)

        p_N = prior_probs['Normal']

        wn_cov = label_train_covariances['Normal']
        det_wn_cov = np.linalg.det(wn_cov)
        p_s_star_given_N = (1 / (2 * (np.sqrt((np.pi ** m) * det_wn_cov)))) * np.exp(-0.5 * CL_value)

        p_Ci_x_p_s_given_Ci_list = []
        for class_label in lab_sorted:
            p_Ci_x_p_s_given_Ci = prior_probs[class_label] * instance_probs[class_label]
            p_Ci_x_p_s_given_Ci_list.append(p_Ci_x_p_s_given_Ci)

        p_s = sum(p_Ci_x_p_s_given_Ci_list)

        Lp = (p_N * p_s_star_given_N) / p_s
        Lp_list.append(Lp)

        p_Ci_x_p_s_Ci_dict = {}
        for class_label in lab_sorted:
            p_Ci = prior_probs[class_label]
            p_s_Ci = instance_probs[class_label]
            p_Ci_x_p_s_Ci = p_Ci * p_s_Ci
            p_Ci_x_p_s_Ci_dict[class_label] = p_Ci_x_p_s_Ci

        # Probabilistic boundary
        p_N_given_s = p_Ci_x_p_s_Ci_dict['Normal'] / p_s
        if p_N_given_s >= Lp:
            y_pred.append('Normal')
        else:
            y_pred.append(max(p_Ci_x_p_s_Ci_dict, key=p_Ci_x_p_s_Ci_dict.get))

    return y_test, y_pred


### KERNEL DENSITY ESTIMATION (KDE) ###
def kde_classifier(dataframe, var_list, lab_sorted, Lp_value):
    """
    Performs classification using a KDE-BN approach.

    Args:
    - dataframe (DataFrame): The pandas DataFrame containing the dataset.
    - var_list (list): A list of feature variables to be used for classification.
    - lab_sorted (list): A sorted list of unique class labels in the dataset.
    - Lp (float): Treshold value to identify Normal class (reducing FAR).

    Returns:
    - y_test (list): True class labels for the test data.
    - y_pred (list): Predicted class labels for the test data.
    """
    X = dataframe.loc[:, var_list]
    Y = dataframe['label']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, stratify=Y)

    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    nc = len(lab_sorted)

    prior_probs = {}
    for class_label in lab_sorted:
        prior_p = (x_train[y_train == class_label].shape[0] + 1) / (len(y_train) + nc)
        prior_probs[class_label] = prior_p

    bandwidth = np.linspace(0.1, 2.0, 10)  # Hyperparameter of KDE is optimized automatically
    kde_models = {}
    best_bandwidth = {}
    for label in lab_sorted:
        kde = KernelDensity(kernel='gaussian')
        param_KDE = {'bandwidth': bandwidth}
        grid_search_KDE = GridSearchCV(estimator=kde, param_grid=param_KDE, cv=10)
        x_train_label = x_train[y_train == label]
        grid_search_KDE.fit(x_train_label)
        best_bandwidth[label] = grid_search_KDE.best_params_
        best_kde = grid_search_KDE.best_estimator_
        kde_models[label] = best_kde

    # print('\nBest KDE hyperparameters:', best_bandwidth)

    log_prob = np.array([kde_models[label].score_samples(x_test) for label in lab_sorted]).T  # Density
    probabilities = np.exp(log_prob - log_prob.max(axis=1)[:, np.newaxis])
    prob_df = pd.DataFrame(probabilities, columns=lab_sorted)

    # print('\nProbability dataframe of KDE:')
    # print(prob_df)

    y_pred = []
    Lp = Lp_value  # Set in advance according to the user's tolerance in order to reduce FAR
    wn_index = lab_sorted.index('Normal')
    for index, row_test_data in prob_df.iterrows():
        prodotto = {}
        for col_name, value in row_test_data.items():
            prodotto[col_name] = prior_probs[col_name] * row_test_data[col_name]
        if row_test_data.iloc[wn_index] >= Lp:
            y_pred.append('Normal')
        else:
            indice_massimo = max(prodotto, key=prodotto.get)
            y_pred.append(indice_massimo)
    return y_test, y_pred


### COST-SENSITIVE TREE-AUGMENTED NAIVE BAYES (TAN) ###
def obtain_weighted_cost_list(df_train, cost_ratio_list):
    """
    Obtain the weighted cost for each class label.
    """
    N = len(df_train)
    class_counts = df_train['label'].value_counts().sort_index()
    weight = {}
    for i, class_label in enumerate(class_counts.index):
        weight[class_label] = (cost_ratio_list[i] * N) / (class_counts * cost_ratio_list).sum()
    return weight


def conditional_mutual_info_score_weighted(df_train, xi, xj, label_col, cost_ratios):
    """
    Compute the weighted conditional mutual information I(Xi, Xj | C) for each label C.
    """
    weighted_cmi = 0
    for label in df_train[label_col].unique():
        subset = df_train[df_train[label_col] == label]
        cmi = mutual_info_score(subset[xi], subset[xj])
        weighted_cmi += cmi * cost_ratios[label]
    return weighted_cmi


def construct_max_spanning_tree_weighted(df_train, features, label_col, cost_ratio_list):
    """
    Construct a maximum spanning tree using weighted conditional mutual information as edge weights.
    """
    G = nx.Graph()
    cost_ratios = obtain_weighted_cost_list(df_train, cost_ratio_list)

    # Add edges between features with weighted CMI as weights
    for xi in features:
        for xj in features:
            if xi != xj:
                weight = conditional_mutual_info_score_weighted(df_train, xi, xj, label_col, cost_ratios)
                G.add_edge(xi, xj, weight=weight)

    # Construct maximum spanning tree
    return nx.maximum_spanning_tree(G)


def construct_TAN_weighted(df_train, features, label_col, cost_ratio_list, root_node=None):
    """
    Construct a Tree-Augmented Naive Bayes model.
    """
    # Step 1: Construct the maximum spanning tree
    tree = construct_max_spanning_tree_weighted(df_train, features, label_col, cost_ratio_list)

    # Check if the tree is created successfully
    if tree is None:
        print("Failed to construct the maximum spanning tree.")
        return None

    # Step 2: Orient the edges to form a directed acyclic graph (DAG)
    root_feature = root_node if root_node else features[0]
    dag = nx.bfs_tree(tree, root_feature)

    # Step 3: Connect the class node to all feature nodes
    for node in tree.nodes:
        dag.add_edge(label_col, node)

    return dag


def prior_probs(sorted_labels, df_train, x_train, y_train, cost_ratio_list):
    """
    Compute the prior probabilities for each class label.
    """
    nc = len(sorted_labels)
    class_counts = df_train['label'].value_counts().sort_index()
    weight = obtain_weighted_cost_list(df_train, cost_ratio_list)

    prior_cpt = pd.DataFrame(index=["Prior"], columns=sorted_labels, dtype=float).fillna(0)
    for class_label in sorted_labels:
        prior_p_weighted = (weight[class_label]*x_train[y_train == class_label].shape[0] + 1) /\
                           ((list(weight.values())*class_counts).sum() + nc)
        prior_cpt[class_label] = prior_p_weighted

    return prior_cpt


def root_node_prob(df_train, features, cost_ratio_list, sorted_labels, root_node=None):
    """
    Compute the conditional probability table (CPT) for the root node.
    """
    weight = obtain_weighted_cost_list(df_train, cost_ratio_list)
    root_node = root_node if root_node else features[0]
    root_values = sorted(df_train[root_node].unique())
    cpt_root_node = pd.DataFrame(index=root_values, columns=sorted_labels, dtype=float).fillna(0)
    for label in sorted_labels:
        num_label = len(df_train[df_train['label'] == label])
        for value in root_values:
            num_value_and_label = len(df_train[(df_train[root_node] == value) & (df_train['label'] == label)])
            prob = ((weight[label] * num_value_and_label) + 1) / (weight[label]*num_label + len(root_values))
            cpt_root_node.loc[value, label] = prob
    return cpt_root_node


def get_parents_for_all_nodes(dag, root_node, label_col):
    """
    Get the parent nodes for each node in the TAN structure.
    """
    parents = {}
    for node in dag.nodes():
        # Skip the root node and label/class node since they do not have the same structure of parents
        if node == root_node or node == label_col:
            continue
        # Get the predecessors in the DAG which are the parents of the node
        parents[node] = list(dag.predecessors(node))
    return parents


def create_cpt_for_node(df_train, node, parents, label_col, cost_ratio_list):
    """
    Create a Conditional Probability Table (CPT) for a given node (parent and child) in the TAN structure.
    """
    cost_ratios = obtain_weighted_cost_list(df_train, cost_ratio_list)
    node_values = sorted(df_train[node].unique())
    parent_label_values = sorted(df_train[label_col].unique())
    other_parent = parents[0] if parents[1] == label_col else parents[1]
    other_parent_values = sorted(df_train[other_parent].unique())
    # Create a MultiIndex for the columns with all combinations of parent label values and other parent values
    multi_index = pd.MultiIndex.from_product([parent_label_values, other_parent_values],
                                             names=[label_col, other_parent])
    # Initialize the CPT DataFrame with the MultiIndex and one row for each node value
    cpt = pd.DataFrame(index=node_values, columns=multi_index, dtype=float).fillna(0)
    # Calculate the probabilities for the CPT
    for node_value in node_values:
        for label_value in parent_label_values:
            for other_parent_value in other_parent_values:
                # Filter the dataframe for the current combination of parent values
                df_filtered = df_train[(df_train[label_col] == label_value) & (df_train[other_parent] == other_parent_value)]

                # Number of occurrences for the current node value given the parent values
                num_value_given_parents = len(df_filtered[df_filtered[node] == node_value])

                # Total number of occurrences for the current parent values
                total_given_parents = len(df_filtered)

                # Calculate the probability with Laplace smoothing
                prob = ((cost_ratios[label_value] * num_value_given_parents) + 1) / (cost_ratios[label_value]*total_given_parents + len(node_values))

                # Set the probability in the CPT
                cpt.loc[node_value, (label_value, other_parent_value)] = prob

    # Sort the CPT by its MultiIndex columns and index rows
    cpt.sort_index(axis=0, inplace=True)  # Sort rows
    cpt.sort_index(axis=1, inplace=True)  # Sort columns
    return cpt


def create_all_feature_cpts(df_train, dag, label_col, cost_ratio_list, root_node):
    """
    Create Conditional Probability Tables (CPTs) for all nodes in the TAN structure.
    """
    cpts = {}
    node_parents = get_parents_for_all_nodes(dag, root_node, label_col)

    # Exclude the root node and label from the nodes to process
    nodes_to_process = set(dag.nodes()) - {root_node, label_col}
    parents = {}
    for node in nodes_to_process:
        parents[node] = node_parents[node]
        cpt = create_cpt_for_node(df_train, node, node_parents[node], label_col, cost_ratio_list)
        cpts[node] = cpt
    return cpts


def lookup_CS_cpd_value(cpt, feature_value, label_value, other_parent_value=None):
    """
    Retrieve the conditional probability value from the CPT.
    """
    if other_parent_value is None:
        # If no other parent, assume it's the prior or root CPT
        return cpt.loc[feature_value, label_value]
    else:
        # For non-root nodes with two parents
        return cpt.loc[feature_value, (label_value, other_parent_value)]


def CS_TAN_prediction(df_test, dag, prior_cpt, root_cpt, cpts, sorted_labels, root_node):
    """
    Perform prediction of the class using the Cost-Sensitive Tree-Augmented Naive Bayes model.
    """
    y_pred = []
    for _, row in df_test.iterrows():
        evidence = row.to_dict()
        label_probs = {}
        for label in sorted_labels:
            product = prior_cpt.loc['Prior', label]  # Start with prior probability
            product *= lookup_CS_cpd_value(root_cpt, evidence[root_node], label)  # Multiply with root node probability
            # Multiply with probabilities from other CPTs
            for feature, cpt in cpts.items():
                if feature != root_node:  # Skip root node as it's already considered
                    other_parent = [p for p in dag.predecessors(feature) if p != 'label'][0]
                    product *= lookup_CS_cpd_value(cpt, evidence[feature], label, evidence[other_parent])
            label_probs[label] = product
        # Select the label with the highest product value
        best_label = max(label_probs, key=label_probs.get)
        y_pred.append(best_label)
    return y_pred


def tan_classifier(X, Y, lista_col, sorted_labels, cost_ratio_list):
    """
    Perform classification using the Cost-Sensitive Tree-Augmented Naive Bayes model.

    Args:
    - X (DataFrame): The pandas DataFrame containing the input features.
    - Y (Series): The pandas Series containing the class labels.
    - lista_col (list): A list of feature variables (from X) to be used for classification.
    - sorted_labels (list): A sorted list of unique class labels in the dataset.
    - cost_ratio_list (list): A list of mis-classification cost for each class label. Higher is the cost for a defined class, higher is the priority of the model to classify that class.

    Returns:
    - Tree Structure (Graph): The constructed network for TAN.
    - y_test (list): True class labels for the test data.
    - y_pred (list): Predicted class labels for the test data.
    """
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    df_train = pd.concat([x_train, y_train], axis=1)

    x_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    df_test = pd.concat([x_test, y_test], axis=1)

    CS_TAN_model = construct_TAN_weighted(df_train, lista_col, 'label', cost_ratio_list)
    pos = nx.circular_layout(CS_TAN_model)
    nx.draw(CS_TAN_model, pos, with_labels=True)
    plt.show()  # Display the TAN structure to the user

    prior_p = prior_probs(sorted_labels, df_train, x_train, y_train, cost_ratio_list)
    # print(prior_p)
    root_p = root_node_prob(df_train, lista_col, cost_ratio_list, sorted_labels)
    # print(root_p)
    others_p = create_all_feature_cpts(df_train, CS_TAN_model, 'label', cost_ratio_list, lista_col[0])
    # print(others_p)

    y_pred = CS_TAN_prediction(df_test, CS_TAN_model, prior_p, root_p, others_p, sorted_labels, lista_col[0])

    return y_test, y_pred
