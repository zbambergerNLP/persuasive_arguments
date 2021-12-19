TRAIN = "train"
TEST = "test"

# Checkpoint constants
BERT_BASE_CASED = "bert-base-cased"

# Metrics Constants:
ACCURACY = "accuracy"
PRECISION = "precision"
RECALL = "recall"
F1 = "f1 measure"
CONFUSION_MATRIX = "confusion_matrix"
CLASSIFICATION_REPORT = "classification_report"

# BERT constants
INPUT_IDS = 'input_ids'
TOKEN_TYPE_IDS = 'token_type_ids'
ATTENTION_MASK = 'attention_mask'

# CMV Dataset
CONVOKIT_DATASET_NAME = 'winning-args-corpus'
CMV_DATASET_NAME = 'cmv_delta_prediction.json'
SUCCESS = 'success'
REPLIES = 'replies'
NUM_LABELS = 2

# Argumentative modes for premises
ETHOS = 'ethos'
LOGOS = 'logos'
PATHOS = 'pathos'
ETHOS_LOGOS = 'ethos_logos'
ETHOS_PATHOS = 'ethos_pathos'
LOGOS_PATHOS = 'logos_pathos'
ETHOS_LOGOS_PATHOS = 'ethos_logos_pathos'
PREMISE_EMPTY = ''
PREMISE_INTERPRETATION = 'interpretation'
PREMISE_MODE_TO_INT = {ETHOS: 0,
                       LOGOS: 1,
                       PATHOS: 2,
                       ETHOS_LOGOS: 3,
                       ETHOS_PATHOS: 4,
                       LOGOS_PATHOS: 5,
                       ETHOS_LOGOS_PATHOS: 6,
                       PREMISE_EMPTY: 7,
                       PREMISE_INTERPRETATION: 8}
INITIAL_PREMISE_TYPES_TO_CONSIDER = 3

# Pandas column names
OP_COMMENT = 'op_comment'
REPLY = 'reply'
LABEL = 'label'


# CMV Premise Mode Dataset
PREMISE_TEXT = 'premise_text'
PREMISE_MODE = 'premise_mode'
PREMISE_DIR_PATH_MAPPING = {
    ETHOS: 'ethos_hidden_states',
    LOGOS: 'logos_hidden_states',
    PATHOS: 'pathos_hidden_states',
}