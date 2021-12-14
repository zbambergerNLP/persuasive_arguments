import os
from bs4 import BeautifulSoup
import pandas as pd


XML = "xml"
PREMISE = "premise"
TYPE = "type"

v1_path = 'v1.0'
v2_path = 'v2.0'
cmv_modes_versions = [v1_path, v2_path]

PREMISE_TEXT = "premise_text"
PREMISE_MODE = "premise_mode"

POSITIVE = 'positive'
NEGATIVE = 'negative'
sign_lst = [POSITIVE, NEGATIVE]


def get_cmv_modes_corpus():
    text = []
    label = []
    current_path = os.getcwd()
    for version in cmv_modes_versions:
        for sign in sign_lst:
            thread_directories = os.path.join(current_path, version, sign)
            for file_name in os.listdir(thread_directories):
                if file_name.endswith(XML):
                    with open(os.path.join(thread_directories, file_name), 'r') as f:
                        data = f.read()
                        bs_data = BeautifulSoup(data, XML)
                        premises = bs_data.find_all(PREMISE)
                        for premise in premises:
                            print(premise)
                            text.append(premise.contents[0])
                            label.append(premise.attrs[TYPE])
    return pd.DataFrame({PREMISE_TEXT: text, PREMISE_MODE: label})


if __name__ == "__main__":
    df = get_cmv_modes_corpus()
    print(df.head())
