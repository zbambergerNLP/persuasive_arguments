import json
import pandas as pd


pairs = []
for line in open('train_pair_data.jsonlist', 'r'):
    pairs.append(json.loads(line))

op_text = []
replies = []
labels = []

for pair in pairs:
    n_replies = min(len(pair['negative']['comments']), len(pair['positive']['comments']))
    op_text += 2 * n_replies * [pair['op_text']]

    for i in range(n_replies):
        replies.append(pair['positive']['comments'][i]['body'])

    for i in range(n_replies):
        replies.append(pair['negative']['comments'][i]['body'])

    labels += [1] * n_replies
    labels += [0] * n_replies

df = pd.DataFrame({'op_text': op_text, 'replies': replies, 'labels': labels})

