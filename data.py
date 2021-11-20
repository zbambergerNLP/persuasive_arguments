import numpy as np
import pandas as pd
from convokit import Corpus, download

if __name__ == "__main__":
    corpus = Corpus(filename=download("winning-args-corpus"))
    corpus.print_summary_stats()
    problematic_conversations = {}
    transformer_inputs = pd.DataFrame()
    for conversation_id, utterences in corpus.conversations.items():
        try:
            conversation_paths = utterences.get_longest_paths()
            # print(f'Number of conversation paths: {len(conversation_paths)}')
            for i, path in enumerate(conversation_paths):
                for j, text_entry in enumerate(path):
                    pass
                    # print(f'conversation path #{i}, text #{j}')
                    # print(f'conversation path #{i}, text #{j}, text content: {text_entry.text.strip()}')
        except TypeError as type_error:
            # print(f'TypeError occurred in conversation ID #{conversation_id}: {type_error}')
            problematic_conversations[conversation_id] = type_error
        except ValueError as value_error:
            # print(f'ValueError occurred in conversation ID #{conversation_id}: {value_error}')
            problematic_conversations[conversation_id] = value_error
    print(f'Number of problematic conversations: {len(problematic_conversations.keys())}')
