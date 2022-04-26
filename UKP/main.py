import sys
import os
sys.path.append('..')
import UKP.extract_data as ukp
import CMV_2016_pair.extract_data as cmv
from Main.split import Splitter

import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from feature_extraction import LinguisticFeaturesSetUp, LinguisticFeatures, SemanticFeaturesSetUp, SemanticFeatures
from Team_BART.RandomForestClassifier import *


#################
### Constants ###
#################

ROOT = "ROOT"
UKP_PROMPT = "PROMPT"
UKP_ESSAY = "ESSAY CONTENT"
UKP_PROMPT_LABEL = "PROMPT LABEL"
UKP_ESSAY_LABEL = "ESSAY LABEL"
UKP_LABEL_DIFFERENCE = "LABEL DIFFERENCE"
ESSAY_IND = "ESSAY"

def select_Xy(df, X_columns, y_column=None):
    """
    :return:
        X: 2D array of selected columns in X_columns in the data frame
        y: 1D array of y_column in the data frame
    """
    X = df[X_columns].values
    if y_column is not None:
        y = df[y_column].values
        return X, y
    return X

def get_feature_vector(linguistic_features_setup, semantic_features_setup, X, num_of_features):
    result = np.zeros((len(X), num_of_features))
    for i, r in enumerate(X):
        features = np.zeros(num_of_features)
        j = 0
        for c in r:
            if isinstance(c, float):
                features[-1] = c # must be last feature
                continue
            linguistic_features = LinguisticFeatures(c, linguistic_features_setup)
            num_ling_features = len(linguistic_features.vector)
            features[j: j+num_ling_features] = linguistic_features.vector
            j += num_ling_features
            # semantic_features = SemanticFeatures(c, semantic_features_setup)
            # num_sema_features = len(sema_features.vector)
            # features[j: j+num_ling_features] = linguistic_features.vector
            # j += num_sema_features
        result[i] = features
    return np.array(result)

if __name__ == "__main__":

    # Tree Construction
    ukp_graph = nx.DiGraph()
    cmv_graph = nx.DiGraph()
    ukp_graph.add_node(ROOT)
    cmv_graph.add_node(ROOT)

    # Data-Set Extraction
    ukp_df = ukp.extract_data()
    ukp_df.to_csv("ukp_dataset.csv")
    # cmv_df = cmv.extract_data()

    # Seperate X and y
    # cmv_y = cmv_df.pop(cmv.LABEL).values
    # cmv_X = cmv_df.values

    # Assign Train/Validation/Test sets
    unique_inds = []
    for filename in ukp_df[ESSAY_IND].unique():
        # TODO: create list of inds by removing .txt
        name = filename[:-4]
        unique_inds.append(name)

    ukp_all_text = select_Xy(ukp_df,[UKP_PROMPT, UKP_ESSAY])
    linguistic_features_setup = LinguisticFeaturesSetUp(ukp_all_text)
    # semantic_features_setup = SemanticFeaturesSetUp()
    semantic_features_setup = None

    # Experiment 1: Prompt + Essay + Initial Opinion -> Final Opinion
    ukp_X_1, ukp_y_1 = select_Xy(ukp_df,
                                 [UKP_PROMPT, UKP_ESSAY, UKP_PROMPT_LABEL],
                                 UKP_ESSAY_LABEL)
    print("Experiment 1:")
    print("ukp_X_1 shape:", ukp_X_1.shape)
    print("ukp_y_1 shape:", ukp_y_1.shape)
    ukp_X_1_Tr, ukp_X_1_test, ukp_y_1_train, ukp_y_1_test = train_test_split(
        ukp_X_1,
        ukp_y_1,
        test_size=0.15
    )
    print("\n")

    # Extract features from text
    num_features = 2 * LinguisticFeatures.NUM_FEATURES + 1 #semantic features to be added
    ukp_X_1_Tr_vector = get_feature_vector(linguistic_features_setup, None, ukp_X_1_Tr, num_features)
    ukp_X_1_test_vector = get_feature_vector(linguistic_features_setup, None, ukp_X_1_test, num_features)
    print("ukp_X_1 vector shape:", ukp_X_1_Tr_vector.shape)

    # Run Random Forest Experiment
    n_estimators = [5, 10, 20, 50, 100, 200, 500]
    impt, accuracy = run_experiment(n_estimators, ukp_X_1_Tr_vector, ukp_y_1_train, ukp_X_1_test_vector, ukp_y_1_test)
    print(impt, accuracy)


    # Experiment 2: Prompt + Essay  + Initial Opinion -> Opinion Delta
    ukp_X_2, ukp_y_2 = select_Xy(ukp_df,
                                 [UKP_PROMPT, UKP_ESSAY, UKP_PROMPT_LABEL],
                                 UKP_LABEL_DIFFERENCE)
    print("Experiment 2:")
    print("ukp_X_2 shape:", ukp_X_2.shape)
    print("ukp_y_2 shape:", ukp_y_2.shape)
    ukp_X_2_Tr, ukp_X_2_test, ukp_y_2_train, ukp_y_2_test = train_test_split(
        ukp_X_2,
        ukp_y_2,
        test_size=0.15
    )
    print("\n")
    # Extract features from text
    num_features = 2*LinguisticFeatures.NUM_FEATURES+1 #semantic features to be added
    ukp_X_2_Tr_vector = get_feature_vector(linguistic_features_setup, semantic_features_setup, ukp_X_2_Tr, num_features)
    ukp_X_2_test_vector = get_feature_vector(linguistic_features_setup, semantic_features_setup, ukp_X_2_test, num_features)
    print("ukp_X_2 vector shape:", ukp_X_2_Tr_vector.shape)

    # Run Random Forest Experiment
    n_estimators = [5,10,20,50,100,200,500]
    impt, accuracy = run_experiment(n_estimators, ukp_X_2_Tr_vector, ukp_y_2_train, ukp_X_2_test_vector, ukp_y_2_test)
    print(impt, accuracy)


    # Experiment 3: Prompt + Essay -> Opinion Delta
    ukp_X_3, ukp_y_3 = select_Xy(ukp_df,
                                 [UKP_PROMPT, UKP_ESSAY],
                                 UKP_LABEL_DIFFERENCE)
    print("Experiment 3:")
    print("ukp_X_3 shape:", ukp_X_3.shape)
    print("ukp_y_3 shape:", ukp_y_3.shape)
    ukp_X_3_Tr, ukp_X_3_test, ukp_y_3_train, ukp_y_3_test = train_test_split(
        ukp_X_3,
        ukp_y_3,
        test_size=0.15
    )
    print("\n")
    # Extract features from text
    num_features = 2*LinguisticFeatures.NUM_FEATURES #semantic features to be added
    ukp_X_3_Tr_vector = get_feature_vector(linguistic_features_setup, semantic_features_setup, ukp_X_3_Tr, num_features)
    ukp_X_3_test_vector = get_feature_vector(linguistic_features_setup, semantic_features_setup, ukp_X_3_test, num_features)
    print("ukp_X_3 vector shape:", ukp_X_3_Tr_vector.shape)

    # Run Random Forest Experiment
    n_estimators = [5,10,20,50,100,200,500]
    impt, accuracy = run_experiment(n_estimators, ukp_X_3_Tr_vector, ukp_y_3_train, ukp_X_3_test_vector, ukp_y_3_test)
    print(impt, accuracy)

    # Experiment 4: Prompt + Essay  + Initial Opinion -> (Opinion Delta >= 1)
    ukp_X_4 = select_Xy(ukp_df,
                        [UKP_PROMPT, UKP_PROMPT_LABEL, UKP_ESSAY])
    ukp_y_4 = np.greater_equal(ukp_df[UKP_LABEL_DIFFERENCE].values, 1)
    print("Experiment 4:")
    print("ukp_X_4 shape:", ukp_X_4.shape)
    print("ukp_y_4 shape:", ukp_y_4.shape)

    ukp_X_4_Tr, ukp_X_4_test, ukp_y_4_train, ukp_y_4_test = train_test_split(
        ukp_X_4,
        ukp_y_4,
        test_size=0.15
    )
    print("\n")

    # Experiment 5: Prompt + Essay -> (Opinion Delta  >= 1)
    ukp_X_5 = select_Xy(ukp_df,
                        [UKP_PROMPT, UKP_ESSAY])
    ukp_y_5 = np.greater_equal(ukp_df[UKP_LABEL_DIFFERENCE].values, 1)
    print("Experiment 5:")
    print("ukp_X_5 shape:", ukp_X_5.shape)
    print("ukp_y_5 shape:", ukp_y_5.shape)
    ukp_X_5_Tr, ukp_X_5_test, ukp_y_5_train, ukp_y_5_test = train_test_split(
        ukp_X_5,
        ukp_y_5,
        test_size=0.15
    )
    print("\n")

    # Experiment 6: Prompt + Essay  + Initial Opinion -> (Opinion Delta >= 2)
    ukp_X_6 = select_Xy(ukp_df,
                        [UKP_PROMPT, UKP_ESSAY, UKP_PROMPT_LABEL])

    ukp_y_6 = np.greater_equal(ukp_df[UKP_LABEL_DIFFERENCE].values, 2)
    ukp_X_6_Tr, ukp_X_6_test, ukp_y_6_train, ukp_y_6_test = train_test_split(
        ukp_X_6,
        ukp_y_6,
        test_size=0.15
    )

    # Experiment 7: Prompt + Essay -> (Opinion Delta >= 2)
    ukp_X_7 = select_Xy(ukp_df, [UKP_PROMPT, UKP_ESSAY])
    ukp_y_7 = np.greater_equal(ukp_df[UKP_LABEL_DIFFERENCE].values, 2)
    ukp_X_7_Tr, ukp_X_7_test, ukp_y_7_train, ukp_y_7_test = train_test_split(
        ukp_X_7,
        ukp_y_7,
        test_size=0.15
    )




