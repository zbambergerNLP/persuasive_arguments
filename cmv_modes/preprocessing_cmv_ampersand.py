import os
from bs4 import BeautifulSoup
import pandas as pd
import constants


XML = "xml"
PREMISE = "premise"
TYPE = "type"

v1_path = os.path.join('cmv_modes', 'change-my-view-modes-master', 'v1.0')
v2_path = os.path.join('cmv_modes', 'change-my-view-modes-master', 'v2.0')

# TODO(zbamberger): Add support for CMV v2 by parsing .ann files as well as .xml files.
cmv_modes_versions = [v1_path]

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
                            text.append(premise.contents[0])
                            label.append(premise.attrs[TYPE])
    return pd.DataFrame({constants.PREMISE_TEXT: text, constants.PREMISE_MODE: label})
