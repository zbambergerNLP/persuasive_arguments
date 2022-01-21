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
    Return a collection of evaluation metrics given a (logits, labels) pair for a binary classification problem.

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
    """
    Return a collection of evaluation metrics given a (logits, labels) pair for a multi-class classification problem.
    :param eval_pred: A 2-tuple of the form [logits, labels]. Labels is a collection of integers representing the label
        of the input. Logits is a collection of tensors corresponding to the model's logits for each input in the batch.
    :return: A dictionary of metrics containing the following keys: precision, recall, f1, accuracy.
    """
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
    """Evaluate the performance of an n-gram language model on a probing task.

    :param corpus_df: A pandas DataFrame instance with the columns {OP_COMMENT, REPLY, LABEL}. The OP_COMMENT
        and REPLY columns consist of text entries while LABEL is {0, 1}.
    :return: A dictionary of metrics containing the following keys: precision, recall, f1, accuracy.
    """
    corpus_df['combined'] = corpus_df['context_text'] + corpus_df[constants.PREMISE_TEXT]
    train, test = train_test_split(corpus_df, test_size=0.2)

    cv = CountVectorizer(binary=True, min_df=1, max_df=0.95, ngram_range=(1, 2))
    cv.fit_transform(train['combined'].values.astype('U'))
    train_feature_set = cv.transform(train['combined'].values.astype('U'))
    test_feature_set = cv.transform(test['combined'].values.astype('U'))

    lr = LogisticRegression(class_weight='balanced', max_iter=1000)
    y_train = train[constants.PREMISE_MODE]
    lr.fit(train_feature_set, y_train)

    y_pred = lr.predict(test_feature_set)
    y_test = test[constants.PREMISE_MODE]

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    accuracy = sum(y_test == y_pred) / len(y_test)

    return {
        constants.PRECISION: precision,
        constants.RECALL: recall,
        constants.F1: f1,
        constants.ACCURACY: accuracy
    }
