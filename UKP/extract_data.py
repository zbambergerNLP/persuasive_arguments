import numpy as np
import pandas as pd
import os
import csv
import sklearn


#################
### CONSTANTS ###
#################

UKP = "UKP"
DATA_SET_LOC = "brat-project-final/ready"
PROMPTS_FILE_NAME = "../../../Downloads/Argument_Persuasion_Research-master/UKP/prompts.csv"
LABELS_FILE_NAME = "../../../Downloads/Argument_Persuasion_Research-master/UKP/processed_results.csv"
UKP_PROMPTS_SEP = ";"
UKP_PROMPTS_ENCODING = "windows-1252"
ADR = os.path.join("../../../Downloads/Argument_Persuasion_Research-master", UKP)


def extract_data():
    """

    :return: A data frame for the UKP dataset
    """
    cwd = os.getcwd()
    os.chdir(ADR)
    # print(os.listdir(os.getcwd()))
    df = pd.read_hdf('../../../Downloads/Argument_Persuasion_Research-master/UKP/ukp_all.h5', 'df')
    os.chdir(cwd)
    return df


def extract_essays(existing_df):
    """
    Add essays to existing Pandas data-frame.
    Extract these from brat-final-project directory.
    Essays are .txt files and their names can be found in ESSAY_NAME columns.

    :param existing_df: An existing pandas data frame for UKP missing the essays column
    Current columns are:
                            | ID | ESSAY | PROMPT
    :return: pandas data frame with
                            | ID | ESSAY | PROMPT | ESSAY CONTENT |
    columns.
    """
    cwd = os.getcwd()
    os.chdir(DATA_SET_LOC)
    count = 0
    for val in existing_df.ix[:, 'ESSAY']:
        name = val
        try:
            with open(name, 'r', encoding='UTF-8') as essayfile:
                existing_df.ix[count, 'ESSAY CONTENT'] = essayfile.read()
        except:
            pass
        count = count + 1
    os.chdir(cwd)

    return existing_df


def extract_labels(existing_df):
    """
    Add labels to existing Pandas data-frame.

    :param existing_df: An existing pandas data frame for UKP missing the label column
    Current columns are:
                            ESSAY | PROMPT | ESSAY CONTENT |
    :return: pandas data frame with
                            ESSAY | PROMPT | ESSAY CONTENT | PROMPT LABEL | ESSAY LABEL | LABEL DIFFERENCE
    columns.
    """
    cwd = os.getcwd()
    os.chdir(DATA_SET_LOC)
    indices = []
    essay_dict = {}
    for i, val in enumerate(existing_df.ix[:, 'ESSAY']):
        name = val
        if os.path.isfile(name):
            indices.append(i)
            essay_dict[name[:-4]] = i
    os.chdir(cwd)

    lsts = []
    df = pd.DataFrame(columns=existing_df.columns)
    with open(LABELS_FILE_NAME, 'r', encoding='UTF-8') as labelfile:
        reader = csv.reader(labelfile)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            essayID = row[0]
            essayName = "essay"+("000" + str(essayID))[-3:]
            idx = essay_dict[essayName]
            dfrow = existing_df.loc[idx]
            df = df.append(dfrow, ignore_index=True)
            df.ix[i-1, 'PROMPT LABEL'] = int(row[2])
            df.ix[i-1, 'ESSAY LABEL'] = int(row[4])
            df.ix[i-1, 'LABEL DIFFERENCE'] = int(row[4]) - int(row[2]) * int(row[5])
    os.chdir(cwd)
    return df


def split_data(existing_df):
    """
    Create new dataframe with train, test, validation labels and delta value
        70% train, 15% validation, 15% test

    :param existing_df: An existing pandas data frame for UKP missing the label column
    Current columns are:
                        ESSAY | PROMPT | ESSAY CONTENT | PROMPT LABEL | ESSAY LABEL | LABEL DIFFERENCE
    :return: pandas data frame with
                        ID | SET | DELTA
    columns.
    """
    split_df = existing_df.copy()
    split_df = split_df.drop(['PROMPT', 'ESSAY CONTENT'], axis=1)

    unique_essays = sklearn.utils.shuffle(split_df['ESSAY'].unique())
    train_essays = unique_essays[:231] # 231 out of 331
    test_essays = unique_essays[231:281] # 50 out of 331
    validation_essays = unique_essays[281:] # 50 out of 331

    for essay in train_essays:
        split_df.loc[split_df.ESSAY == essay, 'SET'] = "TRAIN"
    for essay in test_essays:
        split_df.loc[split_df.ESSAY == essay, 'SET'] = "TEST"
    for essay in validation_essays:
        split_df.loc[split_df.ESSAY == essay, 'SET'] = "VALIDATION"

    split_df = split_df[["ESSAY","SET","PROMPT LABEL", "ESSAY LABEL", "LABEL DIFFERENCE"]]
    split_df = split_df.rename(index=str, columns={"ESSAY": "ID", "LABEL DIFFERENCE": "DELTA"})
    split_df['ID'] = split_df['ID'].apply(lambda x: x.replace(".txt", ""))

    return split_df
    

if __name__ == "__main__":
    df = extract_data()
    # Only run once or will produce new train, test, validation sets
    split = split_data(df)
    split.loc[split.SET == "TRAIN"].to_csv("train_only.csv", index=False)
    split.loc[split.SET == "VALIDATION"].to_csv("validation_only.csv", index=False)
    split.loc[split.SET == "TEST"].to_csv("test_only.csv", index=False)
    split.to_csv("all_labeled_data.csv", index=False)
