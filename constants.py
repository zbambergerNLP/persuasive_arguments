import os
import pathlib

TRAIN = "train"
EVAL = "eval"
TEST = "test"
VALIDATION = "validation"


# Flag names
FINE_TUNE_MODEL = 'fine_tune_model'
PROBE_MODEL_ON_PREMISE_MODES = 'probe_model_on_premise_modes'
PROBING_MODEL = 'probing_model'
FINE_TUNE_MODEL_ON_PREMISE_MODES = 'fine_tune_model_on_premise_modes'
FINE_TUNED_MODEL_PATH = 'fine_tuned_model_path'
DATASET_NAME = 'dataset_name'
MODEL_CHECKPOINT_NAME = 'model_checkpoint_name'
NUM_TRAINING_EPOCHS = 'num_training_ephocs'
OUTPUT_DIR = 'output_dir'
LOGGING_DIR = 'logging_dir'
PER_DEVICE_TRAIN_BATCH_SIZE = 'per_device_train_batch_size'
PER_DEVICE_EVAL_BATCH_SIZE = 'per_device_eval_batch_size'
WARMUP_STEPS = 'warmup_steps'
WEIGHT_DECAY = 'weight_decay'
LOGGING_STEPS = 'logging_steps'
GENERATE_NEW_PROBING_DATASET = 'generate_new_probing_dataset'

# Checkpoint constants
BERT_BASE_CASED = "bert-base-cased"

# Metrics Constants:
EPOCH = "epoch"
LOSS = "loss"
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
CMV= 'CMV'
BINARY_CMV_DELTA_PREDICTION = 'binary_cmv_delta_prediction'
CONVOKIT_DATASET_NAME = 'winning-args-corpus'
CMV_DATASET_NAME = 'cmv_delta_prediction.json'
SUCCESS = 'success'
REPLIES = 'replies'
NUM_LABELS = 2
FIRST_TEXT = 'first_text'
SECOND_TEXT = 'second_text'

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

# Intra-Argument Relations
INTRA_ARGUMENT_RELATIONS = 'intra_argument_relations'
SENTENCE_1 = 'sentence_1'
SENTENCE_2 = 'sentence_2'
PREPOSITION_DISTANCE = 'preposition_distance'

# TODO: Remove PREMISE_INTERPRETATION from the below dictionary of supported premise types.
PREMISE_MODE_TO_INT = {ETHOS: 0,
                       LOGOS: 1,
                       PATHOS: 2,
                       ETHOS_LOGOS: 3,
                       ETHOS_PATHOS: 4,
                       LOGOS_PATHOS: 5,
                       ETHOS_LOGOS_PATHOS: 6}
INITIAL_PREMISE_TYPES_TO_CONSIDER = 3

# Pandas column names
OP_COMMENT = 'op_comment'
LABEL = 'label'


# CMV Premise Mode Dataset
BINARY_PREMISE_MODE_PREDICTION = 'binary_premise_mode_prediction'
PREMISE_TEXT = 'premise_text'
CLAIM_TEXT = 'context_text'
PREMISE_MODE = 'premise_mode'
PREMISE_DIR_PATH_MAPPING = {
    ETHOS: 'ethos_hidden_states',
    LOGOS: 'logos_hidden_states',
    PATHOS: 'pathos_hidden_states',
}

# Probe Modeling
PRETRAINED = "pretrained"
FINE_TUNED = "finetuned"
MULTICLASS = "multiclass"
PROBING = "probing"
LOGISTIC_REGRESSION = 'logistic_regression'
MLP = "mlp"
HIDDEN_STATE = "hidden_state"

# File Types
JSON = "json"
PYTORCH = "pt"
PARQUET = "parquet"
XML = "xml"
ANN = "ann"
TXT = "txt"

# Wandb constants
RESULTS = 'results'
LOG = 'log'

# GNN
NODE_DIM = 60
BERT_HIDDEN_DIM = 768
GCN = 'GCN'
SAGE = 'SAGE'
GAT = 'GAT'

# Pre-processing
v1_path = 'v1.0'
v2_path = 'v2.0'
POSITIVE = 'positive'
NEGATIVE = 'negative'
SIGN_LIST = [POSITIVE, NEGATIVE]
OP = "OP"
TITLE = "title"
NODE_ID = "id"
NODE = "node"
NODES = 'nodes'
PREMISE = "premise"
CLAIM = "claim"
TYPE = "type"
REPLY = 'reply'
REFERENCE = 'ref'
RELATION = 'rel'
ID_TO_INDEX = 'id_to_idx'
ID_TO_TEXT = 'id_to_text'
ID_TO_NODE_TYPE = 'id_to_node_type'
ID_TO_NODE_SUB_TYPE = 'id_to_node_sub_type'
INDEX_TO_ID = 'idx_to_id'
EDGES = 'edges'
EDGES_TYPES = 'edges_types'
SUPPORT = 'support'
ATTACK = 'attack'
SUPER_NODE = 'super_node'
initial_range = 0.05


# UKP parsing
UKP = "UKP"

# Dictionary Keys
NAME = "name"
ENTITIES = "entities"
ATTRIBUTES = "attributes"
RELATIONS = "relations"

# Parser related constants
T = "T"
A = "A"
R = "R"

# CSV column names
ID_CSV = "ID"
DELTA_CSV = "DELTA"

###################
###### PATH  ######
###################
if os.name == 'nt':
    if os.getlogin() == "b.noam":
        BASE_DIR = pathlib.PurePath("/home/b.noam/persuasive_arguments")
    else:
        BASE_DIR = pathlib.PurePath("/home/b.noam/persuasive_arguments") #TODO: Zach please fill this
else:
    BASE_DIR = pathlib.PurePath("/home/b.noam/persuasive_arguments") #TODO: Zach please fill this

UKP_DIR = BASE_DIR / 'UKP'
UKP_DATA = UKP_DIR / 'brat-project-final'
UKP_LABELS_FILE = UKP_DIR / 'all_labeled_data.csv'