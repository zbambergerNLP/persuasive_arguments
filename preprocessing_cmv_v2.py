import os
from bs4 import BeautifulSoup
import pandas as pd


path = './change-my-view-modes/v2.0/'

text = []
label = []
sign_lst = ['positive', 'negative']

for sign in sign_lst:
    for file in os.listdir(path + sign):
        if file.endswith('xml'):
            with open(path + sign + '/' + file, 'r') as f:
                data = f.read()
                bs_data = BeautifulSoup(data, 'xml')
                premises = bs_data.find_all('premise')
                for premise in premises:
                    text.append(premise.contents[0])
                    label.append(premise.attrs['type'])

df = pd.DataFrame({'text': text, 'label': label})