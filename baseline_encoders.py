from __future__ import annotations

import argparse
import copy
import math
import os
import random
import typing

import datasets
import torch
import torch.nn.functional as F
import transformers
from sentence_transformers import SentenceTransformer
import numpy as np
import tqdm

import utils
import wandb

import constants
import metrics
from cmv_modes import preprocessing_knowledge_graph
from data_loaders import create_dataloaders_for_k_fold_cross_validation


"""
Example usage: 
srun --gres=gpu:1 -p nlp python baseline_encoders.py \
    --data 'CMV' \
    --model_type 'paragraph' \
    --num_epochs 100 \
    --batch_size 16 \
    --positive_example_weight 1 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --scheduler_gamma 0.99 \
    --test_percent 0.1 \
    --val_percent 0.1 \
    --debug '' \
    --use_k_fold_cross_validation 'False' \
    --num_cross_validation_splits 5 \
    --fold_index 0 \
    --dropout_probability 0.3 \
    --encoder_type 'sbert' \
    --seed 42 
"""

# TODO: Assert that values of string flags are permitted. Perform these checks via a dedicated function.
parser = argparse.ArgumentParser(
    description='Process flags for experiments on processing graphical representations of arguments through GNNs.')
parser.add_argument('--data',
                    type=str,
                    default='CMV',
                    help="Defines which dataset to use. Either CMV or UKP.")
parser.add_argument('--model_type',
                    type=str,
                    default='sentence_pooling',
                    help='One of the following pre-defined types: {paragraph, sentence_pooling, sentence_concat}. '
                         'The paragraph model learns over inputs of the form [CLS, title + OP, SEP, reply, SEP].\n'
                         'The sentence_pooling model computes embeddings for each sentence individually, then pools'
                         'the embedding of each utterance to an example-level representation.\n'
                         'Finally, the sentence_concat model computes sentence embeddings as in sentence_pooling. '
                         'However, in the sentence_concat model, we compute an example-level representation by '
                         'concatenating the representations of the first 10 utterances. ')
parser.add_argument('--max_sentence_length',
                    type=int,
                    default=50,
                    help='The maximum number of tokens which is permitted in a tokenized sentence. Sentences which'
                         'are longer than this length are truncated. Sentences shorter than this length are padded'
                         'to it with PAD tokens.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=100,
                    help="The number of training rounds over the knowledge graph dataset.")
parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    help="The number of examples per batch per device during both training and evaluation.")
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-4,
                    help="The learning rate used by the GCN+BERT model during training.")
parser.add_argument('--positive_example_weight',
                    type=int,
                    default=1,
                    help="The weight given to positive examples in the loss function")
parser.add_argument('--weight_decay',
                    type=float,
                    default=5e-4,
                    help="The weight decay parameter supplied to the optimizer for use during training.")
parser.add_argument('--test_percent',
                    type=float,
                    default=0.1,
                    help='The proportion (ratio) of samples dedicated to the test set.')
parser.add_argument('--val_percent',
                    type=float,
                    default=0.1,
                    help='The proportion (ratio) of samples dedicated to the validation set.')
parser.add_argument('--debug',
                    type=bool,
                    default=False,
                    help="Work in debug mode")
parser.add_argument('--use_k_fold_cross_validation',
                    type=str,
                    default="True",
                    help="True if we intend to perform cross validation on the dataset. False otherwise. Using this"
                         "option is advised if the dataset is small.")
parser.add_argument('--max_num_rounds_no_improvement',
                    type=int,
                    default=10,
                    help="The permissible number of validation set evaluations in which the desired metric does not "
                         "improve. If the desired validation metric does not improve within this number of evaluation "
                         "attempts, then early stopping is performed.")
parser.add_argument('--metric_for_early_stopping',
                    type=str,
                    default=constants.ACCURACY,
                    help="The metric to track on the validation set as part of early stopping. If this metric does not "
                         "improve after a few evalution steps on the validation set, we perform early stopping.")
parser.add_argument('--num_cross_validation_splits',
                    type=int,
                    default=5,
                    help="The number of cross validation splits we perform as part of k-fold cross validation.")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="The seed used for random number generation and sampling.")
parser.add_argument('--scheduler_gamma',
                    type=float,
                    default=0.9,
                    help="Gamma value used for the learning rate scheduler during training.")
parser.add_argument('--dropout_probability',
                    type=float,
                    default=0.3,
                    help="The dropout probability across each layer of the MLP in the baseline model.")
parser.add_argument('--fold_index',
                    type=int,
                    default=0,
                    help="The partition index of the held out data as part of k-fold cross validation.")
parser.add_argument('--encoder_type',
                    type=str,
                    default='sbert',
                    help="The model used to both tokenize and encode the textual context of argumentative "
                         "prepositions.")


class SentenceEncoderBaseline(torch.nn.Module):
    def __init__(self,
                 use_frozen_encoder: bool,
                 hidden_channels: typing.List[int],
                 device: torch.device,
                 output_dim: int = 2,
                 encoder_type: str = "bert",
                 dropout_prob: float = 0):
        """
        Initialize a persuasiveness prediction baseline model.

        Initialize a transformer encoder model (followed by an MLP) used to predict argument persuasiveness given
        tokenized sequential inputs (utterances from the CMV posts's title, OP, and the persuasive reply).
        Recall that within CMV, argument context refers to the title and OP text, and the argument corresponds to the
        persuasive reply (which may or may not have earned a Delta).

        :param use_frozen_encoder: True we we intend to freeze the BERT encoder which precedes the MLP. False otherwise.
        :param pooling_type: The string name depicting the pooling aggregation method. One of {"max_pooling",
            "average_pooling"}. For the pooling options, all sentence embeddings are pooled to form the input for the MLP.
        :param device: The device on which the model and data are stored.
        :param encoder_type: The name of the encoder model used to create a representation of inputted arguments.
        """
        super(SentenceEncoderBaseline, self).__init__()
        self.encoder_type = encoder_type

        # Initialize appropriate encoder.
        if self.encoder_type == "bert":
            self._encoder_model = transformers.BertForSequenceClassification.from_pretrained(
                constants.BERT_BASE_CASED,
                num_labels=constants.NUM_LABELS).to(device)
        elif self.encoder_type == "sbert":
            self._encoder_model = SentenceTransformer("all-distilroberta-v1").to(device)
        else:
            raise RuntimeError(f"Unsupported encoder type: {self.encoder_type}")

        if use_frozen_encoder:
            for param in self._encoder_model.parameters():
                param.requires_grad = False

        # Create a pooled representation for sentences.
        self.pooling = torch.nn.AdaptiveMaxPool1d(output_size=1, return_indices=False)

        # Create a MLP to process pooled representations for sentences.
        prev_layer_dimension = constants.BERT_HIDDEN_DIM
        self.mlp = torch.nn.ModuleList().to(device)
        self.dropout_prob = dropout_prob
        for layer_dim in hidden_channels[1:]:
            linear_layer = torch.nn.Linear(prev_layer_dimension, layer_dim)
            prev_layer_dimension = layer_dim
            self.mlp.append(linear_layer)
            self.mlp.append(torch.nn.ReLU())
            self.mlp.append(torch.nn.Dropout(p=dropout_prob))
        self.mlp.append(torch.nn.Linear(prev_layer_dimension, output_dim))
        self.device = device
        
    def forward(self, encoder_inputs):
        """Pass tokenized inputs through the sentence baseline model as part of the forward pass.

        :param encoder_inputs: A dictionary mapping string keys to tensor values corresponding to encoder inputs. Each
            input tensor has shape [batch_size x sequence_length, 1]
        :return: A tensor with shape [batch_size, num_labels].
        """
        # Compute hidden representations for context + argument pairs.
        if self.encoder_type == "bert":
            pooled_embeddings = []
            for example in encoder_inputs:
                encoder_outputs = self._encoder_model.forward(
                    input_ids=example[constants.INPUT_IDS],
                    attention_mask=example[constants.ATTENTION_MASK],
                    token_type_ids=example[constants.TOKEN_TYPE_IDS],
                    output_hidden_states=True,
                )
                sequence_embeddings = encoder_outputs.hidden_states[-1].to(self.device)[:, 0, :]
                pooled_embeddings.append(
                    self.pooling(
                        torch.transpose(
                            sequence_embeddings,
                            dim0=0,
                            dim1=1,
                        )
                    ).flatten()
                )
        else:  # SBERT model is used, which means that the model does not accept token type IDs.
            pooled_embeddings = []
            for example in encoder_inputs:
                sequence_embeddings = self._encoder_model({
                    constants.INPUT_IDS: example[constants.INPUT_IDS],
                    constants.ATTENTION_MASK: example[constants.ATTENTION_MASK]})['sentence_embedding']
                pooled_embeddings.append(
                    self.pooling(
                        torch.transpose(
                            sequence_embeddings,
                            dim0=0,
                            dim1=1,
                        )
                    ).flatten()
                )
        intermediate_value = torch.stack(pooled_embeddings)
        for layer_index, layer in enumerate(self.mlp):
            intermediate_value = layer(intermediate_value)
        label_probabilities = F.softmax(intermediate_value, dim=1)
        return label_probabilities

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            validation_loader: torch.utils.data.DataLoader,
            num_labels: int,
            weight: torch.Tensor,
            num_epochs: int,
            optimizer: torch.optim.Optimizer,
            experiment_name: str,
            max_num_rounds_no_improvement: int,
            metric_for_early_stopping: str,
            scheduler: typing.Union[torch.optim.lr_scheduler.ExponentialLR,
                                    torch.optim.lr_scheduler.ConstantLR] = None):
        """
        Train the baseline model on the persuasiveness prediction task.

        :param train_loader: A 'torch.utils.data.DataLoader' wrapping a 'preprocessing.CMVDataset' instance. This loader
            contains the training set.
        :param validation_loader: A 'torch.utils.data.DataLoader` wrapping a `preprocessing.CMVDataset` instance. This
            loader contains the validation set.
        :param num_labels: The number of labels for the probing classification problem.
        :param loss_function: A 'torch.nn' loss instance such as 'torch.nn.BCELoss'.
        :param num_epochs: The number of epochs used to train the probing model on the probing dataset.
        :param optimizer: A 'torch.optim' optimizer instance (e.g., SGD).
        :param scheduler: A `torch.optim.lr_scheduler` instance used to adjust the learning rate of the optimizer.
        :param max_num_rounds_no_improvement: The maximum number of iterations over the validation set in which accuracy
            does not increase. If validation accuracy does not increase within this number of loops, we stop training
            early.
        :param metric_for_early_stopping: The metric used to determine whether or not to stop early. If the metric of
            interest does not improve within `max_num_rounds_no_improvement`, then we stop early.
        """
        self.train().to(device)
        highest_accuracy = 0
        lowest_loss = math.inf
        num_rounds_no_improvement = 0
        epoch_with_optimal_performance = 0
        best_model_dir_path = os.path.join(os.getcwd(), 'tmp')
        utils.ensure_dir_exists(best_model_dir_path)
        best_model_path = os.path.join(
            best_model_dir_path,
            f'optimal_{metric_for_early_stopping}_{experiment_name}.pt')
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            train_acc = 0.0
            train_num_batches = 0
            for data in tqdm.tqdm(train_loader):
                optimizer.zero_grad()
                targets = torch.tensor([example[constants.LABEL][0] for example in data]).to(device)
                encoder_inputs = []
                for example in data:
                    input_dict = {constants.INPUT_IDS: torch.tensor(example[constants.INPUT_IDS]).to(device),
                                  constants.ATTENTION_MASK: torch.tensor(example[constants.ATTENTION_MASK]).to(device)}
                    if self.encoder_type == 'bert':
                        input_dict[constants.TOKEN_TYPE_IDS] = torch.tensor(example[constants.TOKEN_TYPE_IDS]).to(device)
                    encoder_inputs.append(input_dict)
                outputs = self(encoder_inputs).float().to(device)
                preds = torch.argmax(outputs, dim=1).to(device)
                loss = F.cross_entropy(
                    outputs,
                    target=F.one_hot(targets, num_labels).float().to(device),
                    weight=weight.to(device),
                )
                loss.backward()
                optimizer.step()
                num_correct_preds = (preds == targets).sum().float()
                accuracy = num_correct_preds / targets.shape[0] * 100
                train_num_batches += 1
                train_loss += loss.item()
                train_acc += accuracy
            learning_rate = scheduler.get_last_lr()[0]
            scheduler.step()
            wandb.log({
                f"{constants.TRAIN} {constants.ACCURACY}": train_acc / train_num_batches,
                f"{constants.TRAIN} {constants.EPOCH}": epoch,
                f"{constants.TRAIN} {constants.LOSS}": train_loss / train_num_batches,
                "learning rate": learning_rate
            })

            # Perform Evaluation
            self.eval()
            with torch.no_grad():
                validation_loss = 0.0
                validation_acc = 0.0
                validation_num_batches = 0
                for data in tqdm.tqdm(validation_loader):
                    targets = torch.tensor([example[constants.LABEL][0] for example in data]).to(device)
                    encoder_inputs = []
                    for example in data:
                        input_dict = {constants.INPUT_IDS: torch.tensor(example[constants.INPUT_IDS]).to(device),
                                      constants.ATTENTION_MASK: torch.tensor(example[constants.ATTENTION_MASK]).to(
                                          device)}
                        if self.encoder_type == 'bert':
                            input_dict[constants.TOKEN_TYPE_IDS] = torch.tensor(example[constants.TOKEN_TYPE_IDS]).to(
                                device)
                        encoder_inputs.append(input_dict)
                    outputs = self(encoder_inputs).float().to(device)
                    preds = torch.argmax(outputs, dim=1).to(device)
                    loss = F.cross_entropy(
                        outputs,
                        target=F.one_hot(targets, num_labels).float().to(device),
                        weight=weight.to(device),
                    )
                    validation_loss += loss.item()
                    num_correct_preds = (preds == targets).sum().float()
                    accuracy = num_correct_preds / targets.shape[0] * 100
                    validation_acc += accuracy
                    validation_num_batches += 1
                validation_loss = validation_loss / validation_num_batches
                validation_acc = validation_acc / validation_num_batches
                wandb.log({f"{constants.VALIDATION} {constants.ACCURACY}": validation_acc,
                           f"{constants.VALIDATION} {constants.EPOCH}": epoch,
                           f"{constants.VALIDATION} {constants.LOSS}": validation_loss})
                if metric_for_early_stopping == constants.LOSS and validation_loss < lowest_loss:
                    lowest_loss = validation_loss
                    num_rounds_no_improvement = 0
                    epoch_with_optimal_performance = epoch
                    torch.save(self.state_dict(), best_model_path)
                elif metric_for_early_stopping == constants.ACCURACY and validation_acc > highest_accuracy:
                    highest_accuracy = validation_acc
                    num_rounds_no_improvement = 0
                    epoch_with_optimal_performance = epoch
                    torch.save(self.state_dict(), best_model_path)
                else:
                    num_rounds_no_improvement += 1
                if num_rounds_no_improvement == max_num_rounds_no_improvement:
                    print(f'Performing early stopping after {epoch} epochs.\n')
                    self.load_state_dict(torch.load(best_model_path))
                    break

        print(f'Optimal model obtained from epoch #{epoch_with_optimal_performance}')
        self.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)
    
    def evaluate(self, dataloader, split_name):
        self.eval()
        self.to(device)
        with torch.no_grad():
            preds_list = []
            targets_list = []
            for batch in tqdm.tqdm(dataloader):
                targets = torch.tensor([example[constants.LABEL][0] for example in batch]).to(device)
                encoder_inputs = []
                for example in batch:
                    input_dict = {constants.INPUT_IDS: torch.tensor(example[constants.INPUT_IDS]).to(device),
                                  constants.ATTENTION_MASK: torch.tensor(example[constants.ATTENTION_MASK]).to(
                                      device)}
                    if self.encoder_type == 'bert':
                        input_dict[constants.TOKEN_TYPE_IDS] = torch.tensor(example[constants.TOKEN_TYPE_IDS]).to(
                            device)
                    encoder_inputs.append(input_dict)
                outputs = self(encoder_inputs).float().to(device)
                preds = torch.argmax(outputs, dim=1).cpu()
                preds_list.append(preds)
                targets_list.append(targets.cpu())
            preds_list = np.concatenate(preds_list)
            targets_list = np.concatenate(targets_list)
            eval_metrics = metrics.compute_metrics(num_labels=constants.NUM_LABELS,
                                                   preds=preds_list,
                                                   targets=targets_list,
                                                   split_name=split_name)
            for metric_name, metric_value in eval_metrics.items():
                wandb.summary[f"eval_{metric_name}"] = metric_value
            return eval_metrics


class ParagraphEncoderBaseline(torch.nn.Module):
    def __init__(self,
                 use_frozen_encoder: bool,
                 mlp_layers: torch.nn.Sequential,
                 device: torch.device,
                 encoder_type: str = "bert"):
        """
        Initialize a persuasiveness prediction baseline model.

        Initialize a transformer encoder model (followed by an MLP) used to predict argument persuasiveness given
        tokenized sequential inputs of the form [argument context, argument]. Within CMV, argument context refers to
        the title and OP text, and the argument corresponds to the persuasive reply (which may or may not have earned a
        Delta).

        :param use_frozen_encoder: True we we intend to freeze the BERT encoder which precedes the MLP. False otherwise.
        :param mlp_layers: The torch.nn layers corresponding to the layers of the baseline's NLP.
        :param device: The device on which the model and data are stored.
        :param encoder_type: The name of the encoder model used to create a representation of inputted arguments.
        """
        super(ParagraphEncoderBaseline, self).__init__()
        self.encoder_type = encoder_type

        # Initialize appropriate encoder.
        if self.encoder_type == "bert":
            self._encoder_model = transformers.BertForSequenceClassification.from_pretrained(
                constants.BERT_BASE_CASED,
                num_labels=constants.NUM_LABELS).to(device)
        elif self.encoder_type == "sbert":
            self._encoder_model = SentenceTransformer("all-distilroberta-v1").to(device)
        else:
            raise RuntimeError(f"Unsupported encoder type: {self.encoder_type}")

        if use_frozen_encoder:
            for param in self._encoder_model.parameters():
                param.requires_grad = False
        self._mlp = mlp_layers.to(device)
        self.device = device

    def forward(self, encoder_inputs):
        """Pass tokenized inputs through the model as part of the forward pass.

        :param encoder_inputs: A dictionary mapping string keys to tensor values corresponding to encoder inputs. Each
            input tensor has shape [batch_size, sequence_length]
        :return: A tensor with shape [batch_size, num_labels].
        """
        # Compute hidden representations for context + argument pairs.
        if self.encoder_type == "bert":
            encoder_outputs = self._encoder_model.forward(
                input_ids=encoder_inputs[constants.INPUT_IDS],
                attention_mask=encoder_inputs[constants.ATTENTION_MASK],
                token_type_ids=encoder_inputs[constants.TOKEN_TYPE_IDS],
                output_hidden_states=True,
            )
            sequence_embeddings = encoder_outputs.hidden_states[-1].to(self.device)[:, 0, :]
        else:
            sequence_embeddings = self._encoder_model({
                constants.INPUT_IDS: encoder_inputs[constants.INPUT_IDS],
                constants.ATTENTION_MASK: encoder_inputs[constants.ATTENTION_MASK]})['sentence_embedding']

        logits = self._mlp(sequence_embeddings)
        label_probabilities = F.softmax(logits, dim=1)
        return label_probabilities

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            validation_loader: torch.utils.data.DataLoader,
            num_labels: int,
            weight: torch.Tensor,
            num_epochs: int,
            optimizer: torch.optim.Optimizer,
            experiment_name: str,
            max_num_rounds_no_improvement: int,
            metric_for_early_stopping: str,
            scheduler: typing.Union[torch.optim.lr_scheduler.ExponentialLR,
                                    torch.optim.lr_scheduler.ConstantLR] = None):
        """
        Train the baseline model on the persuasiveness prediction task.

        :param train_loader: A 'torch.utils.data.DataLoader' wrapping a 'preprocessing.CMVDataset' instance. This loader
            contains the training set.
        :param validation_loader: A 'torch.utils.data.DataLoader` wrapping a `preprocessing.CMVDataset` instance. This
            loader contains the validation set.
        :param num_labels: The number of labels for the probing classification problem.
        :param loss_function: A 'torch.nn' loss instance such as 'torch.nn.BCELoss'.
        :param num_epochs: The number of epochs used to train the probing model on the probing dataset.
        :param optimizer: A 'torch.optim' optimizer instance (e.g., SGD).
        :param scheduler: A `torch.optim.lr_scheduler` instance used to adjust the learning rate of the optimizer.
        :param max_num_rounds_no_improvement: The maximum number of iterations over the validation set in which accuracy
            does not increase. If validation accuracy does not increase within this number of loops, we stop training
            early.
        :param metric_for_early_stopping: The metric used to determine whether or not to stop early. If the metric of
            interest does not improve within `max_num_rounds_no_improvement`, then we stop early.
        """
        self.train().to(device)
        highest_accuracy = 0
        lowest_loss = math.inf
        num_rounds_no_improvement = 0
        epoch_with_optimal_performance = 0
        best_model_dir_path = os.path.join(os.getcwd(), 'tmp')
        utils.ensure_dir_exists(best_model_dir_path)
        best_model_path = os.path.join(
            best_model_dir_path,
            f'optimal_{metric_for_early_stopping}_{experiment_name}.pt')
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            train_acc = 0.0
            train_num_batches = 0
            for data in tqdm.tqdm(train_loader):
                optimizer.zero_grad()
                targets = data[constants.LABEL].to(device)
                encoder_inputs = {
                    constants.INPUT_IDS: data[constants.INPUT_IDS].to(self.device),
                    constants.ATTENTION_MASK: data[constants.ATTENTION_MASK].to(self.device)
                }
                if self.encoder_type == "bert":
                    encoder_inputs[constants.TOKEN_TYPE_IDS] = data[constants.TOKEN_TYPE_IDS].to(self.device)

                outputs = self(encoder_inputs).float().to(device)
                preds = torch.argmax(outputs, dim=1).to(device)
                loss = F.cross_entropy(
                    outputs,
                    target=F.one_hot(targets, num_labels).float().to(device),
                    weight=weight.to(device),
                )
                loss.backward()
                optimizer.step()
                num_correct_preds = (preds == targets).sum().float()
                accuracy = num_correct_preds / targets.shape[0] * 100
                train_num_batches += 1
                train_loss += loss.item()
                train_acc += accuracy
            learning_rate = scheduler.get_last_lr()[0]
            scheduler.step()
            wandb.log({
                f"{constants.TRAIN} {constants.ACCURACY}": train_acc / train_num_batches,
                f"{constants.TRAIN} {constants.EPOCH}": epoch,
                f"{constants.TRAIN} {constants.LOSS}": train_loss / train_num_batches,
                "learning rate": learning_rate
            })

            # Perform Evaluation
            self.eval()
            with torch.no_grad():
                validation_loss = 0.0
                validation_acc = 0.0
                validation_num_batches = 0
                for data in tqdm.tqdm(validation_loader):
                    targets = data[constants.LABEL].to(device)
                    encoder_inputs = {
                        constants.INPUT_IDS: data[constants.INPUT_IDS].to(self.device),
                        constants.ATTENTION_MASK: data[constants.ATTENTION_MASK].to(self.device)
                    }
                    if self.encoder_type == "bert":
                        encoder_inputs[constants.TOKEN_TYPE_IDS] = data[constants.TOKEN_TYPE_IDS].to(self.device)
                    outputs = self(encoder_inputs).to(device)
                    preds = torch.argmax(outputs, dim=1).to(device)
                    loss = F.cross_entropy(
                        outputs,
                        target=F.one_hot(targets, num_labels).float().to(device),
                        weight=weight.to(device),
                    )
                    validation_loss += loss.item()
                    num_correct_preds = (preds == targets).sum().float()
                    accuracy = num_correct_preds / targets.shape[0] * 100
                    validation_acc += accuracy
                    validation_num_batches += 1
                validation_loss = validation_loss / validation_num_batches
                validation_acc = validation_acc / validation_num_batches
                wandb.log({f"{constants.VALIDATION} {constants.ACCURACY}": validation_acc,
                           f"{constants.VALIDATION} {constants.EPOCH}": epoch,
                           f"{constants.VALIDATION} {constants.LOSS}": validation_loss})
                if metric_for_early_stopping == constants.LOSS and validation_loss < lowest_loss:
                    lowest_loss = validation_loss
                    num_rounds_no_improvement = 0
                    epoch_with_optimal_performance = epoch
                    torch.save(self.state_dict(), best_model_path)
                elif metric_for_early_stopping == constants.ACCURACY and validation_acc > highest_accuracy:
                    highest_accuracy = validation_acc
                    num_rounds_no_improvement = 0
                    epoch_with_optimal_performance = epoch
                    torch.save(self.state_dict(), best_model_path)
                else:
                    num_rounds_no_improvement += 1
                if num_rounds_no_improvement == max_num_rounds_no_improvement:
                    print(f'Performing early stopping after {epoch} epochs.\n')
                    self.load_state_dict(torch.load(best_model_path))
                    break

        print(f'Optimal model obtained from epoch #{epoch_with_optimal_performance}')
        self.load_state_dict(torch.load(best_model_path))
        os.remove(best_model_path)

    def evaluate(self,
                 dataloader: torch.data.DataLoader,
                 split_name: str,
                 device: torch.device) -> typing.Mapping[str, float]:
        """Evaluate a trained model on a dataset contained within the provided data loader.

        :param dataloader: The dataloader instance containing the dataset on which the model is evaluated.
        :param split_name: One of {"train", "validation", "test"}
        :param device: The device
        :return: A dictionary mapping metric names to their float values.
        """
        self.eval()
        self.to(device)
        with torch.no_grad():
            preds_list = []
            targets_list = []
            for batch in tqdm.tqdm(dataloader):
                targets = batch[constants.LABEL].to(device)
                encoder_inputs = {
                    constants.INPUT_IDS: batch[constants.INPUT_IDS].to(self.device),
                    constants.ATTENTION_MASK: batch[constants.ATTENTION_MASK].to(self.device)
                }
                if self.encoder_type == "bert":
                    encoder_inputs[constants.TOKEN_TYPE_IDS] = batch[constants.TOKEN_TYPE_IDS].to(self.device)
                outputs = self(encoder_inputs).to(device)
                preds = torch.argmax(outputs, dim=1).cpu()
                preds_list.append(preds)
                targets_list.append(targets.cpu())
            preds_list = np.concatenate(preds_list)
            targets_list = np.concatenate(targets_list)
            eval_metrics = metrics.compute_metrics(num_labels=constants.NUM_LABELS,
                                                   preds=preds_list,
                                                   targets=targets_list,
                                                   split_name=split_name)
            for metric_name, metric_value in eval_metrics.items():
                wandb.summary[f"eval_{metric_name}"] = metric_value
            return eval_metrics


if __name__ == '__main__':
    args = parser.parse_args()
    args_dict = vars(args)
    for parameter, value in args_dict.items():
        print(f'{parameter}: {value}')
    utils.set_seed(args.seed)
    sentence_level = args.model_type != 'paragraph'

    if args.data == constants.CMV:
        examples = preprocessing_knowledge_graph.create_simple_bert_inputs(
            directory_path=os.path.join(os.getcwd(), 'cmv_modes', 'change-my-view-modes-master'),
            version=constants.v2_path,
            sentence_level=sentence_level)

    else:  # UKP dataset
        examples = preprocessing_knowledge_graph.create_simple_bert_inputs_ukp(sentence_level=sentence_level)
    features = examples[0]
    labels = np.array(examples[1])
    assert len(features) == len(labels), f'We expect the same number of features and labels.\n' \
                                         f'Number of feature entries: {len(examples)}.\n' \
                                         f'Number of label enties: {len(labels)}'
    print(f'positive_labels: {sum(labels == 1)}\n'
          f'negative_labels: {sum(labels == 0)}')

    # Initialize tokenizer and dataset feature columns.

    feature_columns = [constants.INPUT_IDS, constants.ATTENTION_MASK]
    if args.encoder_type == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED)
        feature_columns.append(constants.TOKEN_TYPE_IDS)
    elif args.encoder_type == "sbert":
        tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/all-distilroberta-v1")
    else:
        raise RuntimeError(f"invalid encoder type: {args.encoder_type}")
    columns = copy.deepcopy(feature_columns)
    columns.append(constants.LABEL)

    # Tokenize encoder inputs
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    if sentence_level:
        # Truncate and pad sentences to a fixed, pre-specified length.
        dataset = {column_name: [] for column_name in columns}
        dataset.update({
            'batch_index': [],
            'sequence_index': [],
            constants.LABEL: [],
        })
        index_mapping = {}
        instance_counter = 0
        for batch_index, sequences in enumerate(features):
            tokenized_sequences = tokenizer(
                sequences,
                padding='max_length',
                max_length=args.max_sentence_length,
                truncation=True,
                return_tensors="pt")

            for sequence_index in range(len(tokenized_sequences[constants.INPUT_IDS])):
                dataset['batch_index'].append(batch_index)
                dataset['sequence_index'].append(sequence_index)
                dataset[constants.LABEL].append(labels[batch_index])
                for input_name, input_value in tokenized_sequences.items():  # Add features for language model
                    dataset[input_name].append(input_value[sequence_index])

                # Create a mapping from individual example indices to the corresponding indices within the flattened
                # dataset.
                if batch_index not in index_mapping:
                    index_mapping[batch_index] = []
                index_mapping[batch_index].append(instance_counter)
                instance_counter += 1

        dataset = datasets.Dataset.from_dict(dataset)
        transformers.logging.set_verbosity(verbosity)
    else:
        op_text = [pair[0] for pair in features]
        reply_text = [pair[1] for pair in features]
        tokenized_inputs = tokenizer(op_text, reply_text, padding=True, truncation=True)
        transformers.logging.set_verbosity(verbosity)
        dataset_dict = {input_name: input_value for input_name, input_value in tokenized_inputs.items()}
        dataset_dict[constants.LABEL] = labels
        dataset = datasets.Dataset.from_dict(dataset_dict)
        dataset.set_format(type='torch', columns=columns)

    # TODO: Create an MLP as a function of a desired model size. This should correspond with the size of a parallel
    #  GNN model, and be passed via flag.
    # TODO: Enable configuration of the MLP architecture through flag parameters.
    mlp = torch.nn.Sequential(
        torch.nn.Linear(constants.BERT_HIDDEN_DIM, constants.BERT_HIDDEN_DIM),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout_probability),
        torch.nn.Linear(constants.BERT_HIDDEN_DIM, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(args.dropout_probability),
        torch.nn.Linear(512, constants.NUM_LABELS),
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if sentence_level:
        encoder_baseline = SentenceEncoderBaseline(
            use_frozen_encoder=True,
            hidden_channels=[constants.BERT_HIDDEN_DIM, 512],
            device=device,
            encoder_type=args.encoder_type,
            dropout_prob=args.dropout_probability
        )

    else:
        encoder_baseline = ParagraphEncoderBaseline(
            use_frozen_encoder=True,
            mlp_layers=mlp,
            device=device,
            encoder_type=args.encoder_type,
        )

    if utils.str2bool(args.use_k_fold_cross_validation):
        # TODO: modify the below sections
        num_of_examples = len(set(dataset['batch_index']))
        shuffled_indices = random.sample(range(num_of_examples), num_of_examples)
        train_loader, validation_loader, test_loader = create_dataloaders_for_k_fold_cross_validation(
            dataset,
            dataset_type='language_model',
            num_of_examples=num_of_examples,
            shuffled_indices=shuffled_indices,
            batch_size=args.batch_size,
            val_percent=args.val_percent,
            test_percent=args.test_percent,
            k_fold_index=args.fold_index)

        num_cross_validation_splits = args.num_cross_validation_splits
        train_metrics = []
        validation_metrics = []
        test_metrics = []
        # shards = [dataset.shard(num_cross_validation_splits, i, contiguous=True)
        #           for i in range(num_cross_validation_splits)]
        num_of_examples = len(set(dataset['batch_index']))
        shuffled_indices = random.sample(range(num_of_examples), num_of_examples)
        for validation_set_index in range(num_cross_validation_splits):
            num_of_examples = len(set(dataset['batch_index'])) if sentence_level else dataset.num_rows
            shuffled_indices = random.sample(range(num_of_examples), num_of_examples)
            train_loader, validation_loader, test_loader = create_dataloaders_for_k_fold_cross_validation(
                dataset,
                dataset_type='language_model',
                num_of_examples=num_of_examples,
                shuffled_indices=shuffled_indices,
                batch_size=args.batch_size,
                val_percent=args.val_percent,
                test_percent=args.test_percent,
                k_fold_index=args.fold_index,
                index_mapping=index_mapping,
                sentence_level=sentence_level)
            split_model = copy.deepcopy(encoder_baseline).to(device)
            # validation_and_test_sets = shards[validation_set_index].train_test_split(
            #     args.val_percent / (args.val_percent + args.test_percent))
            # validation_set = validation_and_test_sets[constants.TRAIN]
            # print(f'Validation set ({validation_set_index}) positive labels: '
            #       f'{sum(validation_set[constants.LABEL].numpy() == 1)}')
            # print(f'Validation set ({validation_set_index}) negative labels: '
            #       f'{sum(validation_set[constants.LABEL].numpy() == 0)}')
            # test_set = validation_and_test_sets[constants.TEST]
            # training_set = datasets.concatenate_datasets(
            #     shards[0:validation_set_index] + shards[validation_set_index + 1:]).shuffle()
            model_name, group_name, run_name = utils.create_baseline_run_and_model_names(
                dataset_name=args.data,
                encoder_type=args.encoder_type,
                dropout_probability=args.dropout_probability,
                learning_rate=args.learning_rate,
                seed=args.seed,
                validation_split_index=validation_set_index,
                positive_example_weight=args.positive_example_weight,
                scheduler_gamma=args.scheduler_gamma,
                weight_decay=args.weight_decay,
            )
            run = wandb.init(
                project="persuasive_arguments",
                entity="persuasive_arguments",
                group=group_name,
                config=args,
                name=run_name,
                dir='.')

            # TODO: Create a helper utility function which generates the optimizer and scheduler given the necessary
            #  parameters.
            optimizer = torch.optim.Adam(
                split_model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay)
            # train_loader, validation_loader, test_loader = utils.create_data_loaders(
            #     training_set=training_set,
            #     validation_set=validation_set,
            #     test_set=test_set,
            #     batch_size=args.batch_size)
            split_model.fit(
                train_loader=train_loader,
                validation_loader=validation_loader,
                experiment_name=run_name,
                num_labels=2,
                weight=torch.Tensor([1, args.positive_example_weight]),
                num_epochs=args.num_epochs,
                optimizer=optimizer,
                max_num_rounds_no_improvement=args.max_num_rounds_no_improvement,
                metric_for_early_stopping=constants.ACCURACY,
                scheduler=torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=args.scheduler_gamma
                )
            )
            train_metrics.append(
                split_model.evaluate(
                    dataloader=train_loader,
                    split_name=constants.TRAIN,
                    device=device)
            )
            validation_metrics.append(
                split_model.evaluate(dataloader=validation_loader,
                                     split_name=constants.VALIDATION,
                                     device=device)
            )
            test_metrics.append(
                split_model.evaluate(
                    dataloader=test_loader,
                    split_name=constants.TEST,
                    device=device
                )
            )
            run.finish()

        validation_metric_aggregates = utils.aggregate_metrics_across_splits(validation_metrics)
        train_metric_aggregates = utils.aggregate_metrics_across_splits(train_metrics)
        test_metric_aggregates = utils.aggregate_metrics_across_splits(test_metrics)
        print(f'\n*** Train Metrics: ***')
        train_metric_averages, train_metric_stds = utils.get_metrics_avg_and_std_across_splits(
            metric_aggregates=train_metric_aggregates,
            split_name=constants.TRAIN,
            print_results=True)
        print(f'\n*** Validation Metrics: ***')
        validation_metric_averages, validation_metric_stds = utils.get_metrics_avg_and_std_across_splits(
            metric_aggregates=validation_metric_aggregates,
            split_name=constants.VALIDATION,
            print_results=True)
        print(f'\n*** Test Metrics: ***')
        test_metric_averages, test_metric_stds = utils.get_metrics_avg_and_std_across_splits(
            metric_aggregates=test_metric_aggregates,
            split_name=constants.TEST,
            print_results=True)

    # Train and evaluate given a single fold, specified by a provided flag.
    else:
        model_name, group_name, run_name = utils.create_baseline_run_and_model_names(
            dataset_name=args.data,
            encoder_type=args.encoder_type,
            dropout_probability=args.dropout_probability,
            learning_rate=args.learning_rate,
            seed=args.seed,
            validation_split_index=args.fold_index,
            positive_example_weight=args.positive_example_weight,
            scheduler_gamma=args.scheduler_gamma,
            weight_decay=args.weight_decay,
        )
        wandb.init(
            project="persuasive_arguments",
            entity="persuasive_arguments",
            group=group_name,
            config=args,
            name=run_name,
            dir='.')
        config = wandb.config
        optimizer = torch.optim.Adam(
            encoder_baseline.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        num_of_examples = len(set(dataset['batch_index'])) if sentence_level else dataset.num_rows
        shuffled_indices = random.sample(range(num_of_examples), num_of_examples)
        train_loader, validation_loader, test_loader = create_dataloaders_for_k_fold_cross_validation(
            dataset,
            dataset_type='language_model',
            num_of_examples=num_of_examples,
            shuffled_indices=shuffled_indices,
            batch_size=args.batch_size,
            val_percent=args.val_percent,
            test_percent=args.test_percent,
            k_fold_index=args.fold_index,
            index_mapping=index_mapping,
            sentence_level=sentence_level)
        encoder_baseline.fit(
            train_loader=train_loader,
            validation_loader=validation_loader,
            experiment_name=run_name,
            num_labels=2,
            weight=torch.Tensor([1, args.positive_example_weight]),
            num_epochs=args.num_epochs,
            optimizer=optimizer,
            max_num_rounds_no_improvement=args.max_num_rounds_no_improvement,
            metric_for_early_stopping=args.metric_for_early_stopping,
            scheduler=torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=args.scheduler_gamma
            )
        )
        print(metrics.perform_evaluation_on_splits(
            eval_fn=encoder_baseline.evaluate,
            device=device,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader
        ))
