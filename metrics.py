import numpy as np
from datasets import load_metric
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import constants

accuracy = load_metric(constants.ACCURACY)


def compute_metrics_for_binary_classification(eval_pred):
    """
    Return a collection of evaluation metrics given a (logits, labels) pair.

    :param eval_pred: A 2-tuple of the form [logits, labels]. Labels is a collection of booleans. Logits is a collection
        of tensors corresponding to the model's logits for each input in the batch.
    :return: A dictionary of metrics containing the following keys: precision, recall, f1, accuracy.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=predictions, average='binary')
    metrics = {
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1: f1,
        constants.ACCURACY: accuracy.compute(predictions=predictions, references=labels)
    }
    return metrics


def compute_metrics_for_multi_class_classification(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=predictions, average='micro')
    metrics = {
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1: f1,
        constants.ACCURACY: accuracy.compute(predictions=predictions, references=labels)
    }
    return metrics


def get_baseline_scores(corpus_df):
    corpus_df['combined'] = corpus_df['context_text'] + corpus_df['premise_text']
    train, test = train_test_split(corpus_df, test_size=0.2)

    cv = CountVectorizer(binary=True, min_df=1, max_df=0.95, ngram_range=(1, 2))
    cv.fit_transform(train['combined'].values.astype('U'))
    train_feature_set = cv.transform(train['combined'].values.astype('U'))
    test_feature_set = cv.transform(test['combined'].values.astype('U'))

    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced', random_state=42,
                            max_iter=1000)
    y_train = train['premise_mode']
    lr.fit(train_feature_set, y_train)

    y_pred = lr.predict(test_feature_set)
    y_test = test['premise_mode']

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = sum(y_test == y_pred) / len(y_test)

    metrics = {
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1: f1,
        constants.ACCURACY: accuracy
    }

    return metrics