import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler
from torch_geometric.nn import to_hetero, global_mean_pool, global_max_pool
import torch_geometric.loader as geom_data

import metrics
import utils
import wandb
from data_loaders import CMVKGDataset, CMVKGHetroDataset, CMVKGHetroDatasetEdges
from models import GCNWithBertEmbeddings, GCNHetero, GraphSage, GAT
import constants
import argparse
from tqdm import tqdm


"""
Example command:
srun --gres=gpu:1 -p nlp python3 train_and_eval.py \
    --num_epochs 30 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --weight_decay 5e-4 \
    --gcn_hidden_layer_dim 128 \
    --test_percent 0.1 \
    --val_percent 0.1 \
    --rounds_between_evals 1 \
    --model "GAT" \
    --debug "" \
    --hetro "" \
    --use_max_pooling "" \
    --use_k_fold_cross_validation True \
    --num_cross_validation_splits 5
"""


parser = argparse.ArgumentParser(
    description='Process flags for experiments on processing graphical representations of arguments through GNNs.')
parser.add_argument('--hetro',
                    type=bool,
                    default=False,
                    help="Use heterophilous graphs if true and homophilous if False")
parser.add_argument('--hetero_type',
                    type=str,
                    default='nodes',
                    help="Relevant only if herto is True. Possible values are 'nodes' or 'edges'. If the value is 'nodes' then node type is used, if the value is 'edges' then edge type is used")
parser.add_argument('--num_epochs',
                    type=int,
                    default=5,
                    help="The number of training rounds over the knowledge graph dataset.")
parser.add_argument('--batch_size',
                    type=int,
                    default=4,
                    help="The number of examples per batch per device during both training and evaluation.")
parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-2,
                    help="The learning rate used by the GCN+BERT model during training.")
parser.add_argument('--weight_decay',
                    type=float,
                    default=5e-4,
                    help="The weight decay parameter supplied to the optimizer for use during training.")
parser.add_argument('--gcn_hidden_layer_dim',
                    type=int,
                    default=128,
                    help="The dimensionality of the hidden layer within the GCN component of the GCN+BERT model.")
parser.add_argument('--test_percent',
                    type=float,
                    default=0.1,
                    help='The proportion (ratio) of samples dedicated to the test set.')
parser.add_argument('--val_percent',
                    type=float,
                    default=0.1,
                    help='The proportion (ratio) of samples dedicated to the validation set.')
parser.add_argument('--rounds_between_evals',
                    type=int,
                    default=5,
                    help="An integer denoting the number of epcohs that occur between each evaluation run.")
parser.add_argument('--debug',
                    type=bool,
                    default=False,
                    help="Work in debug mode")
parser.add_argument('--use_max_pooling',
                    type=bool,
                    default=False,
                    help="if True use max pooling in GNN else use average pooling")
parser.add_argument('--model',
                    type=str,
                    default='SAGE',
                    help="chose which model to run with the options are: GCN, GAT , SAGE")
parser.add_argument('--use_k_fold_cross_validation',
                    type=bool,
                    default=False,
                    help="True if we intend to perform cross validation on the dataset. False otherwise. Using this"
                         "option is advised if the dataset is small.")
parser.add_argument('--num_cross_validation_splits',
                    type=int,
                    default=5,
                    help="The number of cross validation splits we perform as part of k-fold cross validation.")

# TODO: Fix documentation across this file.

def find_labels_for_batch(batch_data):
    batch_labels = []
    key = batch_data.node_types[0]
    for b in range(len(batch_data[key].y)):
        batch_labels.append(batch_data[key].y[b][0])
    return torch.tensor(batch_labels)


def train(model: torch.nn.Module,
          training_loader: geom_data.DataLoader,
          validation_loader: geom_data.DataLoader,
          epochs: int,
          optimizer: torch.optim.Optimizer,
          device,
          rounds_between_evals: int = 1,
          hetro: bool = False,
          use_max_pooling: bool = False,
          metric_for_early_stopping: str = constants.ACCURACY,
          max_num_rounds_no_improvement: int = 10) -> torch.nn.Module:
    """Train a GCNWithBERTEmbeddings model on examples consisting of persuasive argument knowledge graphs.

    :param model: A torch module consisting of a BERT model (used to produce node embeddings), followed by a GCN.
    :param training_loader: A torch geometric data loader used to feed batches from the training set to the model.
    :param validation_loader: A torch geometric data loader used to feed batches from the validation set to the model.
    :param epochs: The number of iterations over the training set during model training.
    :param optimizer: The torch optimizer used for weight updates during trianing.
    :param rounds_between_evals: An integer denoting the number of epcohs that occur between each evaluation run.
    :return: A trained model.
    """
    model.train()
    model.to(device)
    highest_accuracy = 0
    lowest_loss = math.inf
    num_rounds_no_improvement = 0
    epoch_with_optimal_performance = 0
    best_model_dir_path = os.path.join(os.getcwd(), 'tmp')
    utils.ensure_dir_exists(best_model_dir_path)
    best_model_path = os.path.join(best_model_dir_path, f'optimal_{metric_for_early_stopping}_probe.pt')
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        for sampled_data in tqdm(training_loader):
            sampled_data.to(device)
            optimizer.zero_grad()
            if hetro:
                y = find_labels_for_batch(batch_data=sampled_data)
                gnn_out = model(sampled_data.x_dict, sampled_data.edge_index_dict)
                out = 0
                for key in gnn_out.keys():
                    if use_max_pooling:
                        gnn_out[key] = global_max_pool(gnn_out[key], sampled_data[key].batch)
                    else:
                        gnn_out[key] = global_mean_pool(gnn_out[key], sampled_data[key].batch)
                    out += gnn_out[key]
                out = F.log_softmax(out, dim=1)
            else:
                y = sampled_data.y
                out = model(sampled_data.x, sampled_data.edge_index, sampled_data.batch)
            preds = torch.argmax(out, dim=1)
            y_one_hot = F.one_hot(y, 2)
            if hetro:
                loss = F.cross_entropy(out.float(), y_one_hot.float().to(device))
            else:
                loss = model.loss(out.float(), y_one_hot.float().to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_correct_preds = (preds == y.to(device)).sum().float()
            accuracy = num_correct_preds / y.shape[0] * 100
            epoch_acc += accuracy
            num_batches += 1
        wandb.log({f'training_{constants.ACCURACY}': epoch_acc / num_batches,
                   f'training_{constants.EPOCH}': epoch,
                   f'training_{constants.LOSS}': epoch_loss / num_batches})

        # Perform evaluation on the validation set.
        if epoch % rounds_between_evals == 0:
            model.eval()
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            model.eval()
            for sampled_data in tqdm(validation_loader):
                sampled_data.to(device)
                if hetro:
                    y = find_labels_for_batch(batch_data=sampled_data)
                    gnn_out = model(sampled_data.x_dict, sampled_data.edge_index_dict)
                    out = 0
                    for key in gnn_out.keys():
                        if use_max_pooling:
                            gnn_out[key] = global_max_pool(gnn_out[key], sampled_data[key].batch)
                        else:
                            gnn_out[key] = global_mean_pool(gnn_out[key], sampled_data[key].batch)
                        out += gnn_out[key]
                    out = F.log_softmax(out, dim=1)
                else:
                    y = sampled_data.y
                    out = model(sampled_data.x, sampled_data.edge_index, sampled_data.batch)
                preds = torch.argmax(out, dim=1)
                y_one_hot = F.one_hot(y, 2)
                if hetro:
                    loss = F.cross_entropy(out.float(), y_one_hot.float().to(device))
                else:
                    loss = model.loss(out.float(), y_one_hot.float().to(device))
                num_correct_preds = (preds == y.to(device)).sum().float()
                accuracy = num_correct_preds / y.shape[0] * 100
                num_batches += 1
                epoch_loss += loss.item()
                epoch_acc += accuracy
            validation_loss = epoch_loss / num_batches
            validation_acc = epoch_acc / num_batches
            wandb.log({f'validation_{constants.ACCURACY}': epoch_acc / num_batches,
                       f'validation_{constants.EPOCH}': epoch,
                       f'validation_{constants.LOSS}': epoch_loss / num_batches})

            if metric_for_early_stopping == constants.LOSS and validation_loss < lowest_loss:
                lowest_loss = validation_loss
                num_rounds_no_improvement = 0
                epoch_with_optimal_performance = epoch
                torch.save(model.state_dict(), best_model_path)
            elif metric_for_early_stopping == constants.ACCURACY and validation_acc > highest_accuracy:
                highest_accuracy = validation_acc
                num_rounds_no_improvement = 0
                epoch_with_optimal_performance = epoch
                torch.save(model.state_dict(), best_model_path)
            else:
                num_rounds_no_improvement += 1

            if num_rounds_no_improvement == max_num_rounds_no_improvement:
                print(f'Performing early stopping after {epoch} epochs.\n'
                      f'Optimal model obtained from epoch #{epoch_with_optimal_performance}')
                model.load_state_dict(torch.load(best_model_path))
                break
            model.train()
    return model


def eval(model: torch.nn.Module,
         dataset: geom_data.DataLoader,
         split_name: str,
         hetro: bool,
         device,
         use_max_pooling: bool = False):
    """
    Evaluate the performance of a GCNWithBertEmbeddings model.

    The test set used for this evaluation consists of distinct examples from those used by the model during training.

    :param model: A torch module consisting of a BERT model (used to produce node embeddings), followed by a GCN.
    :param dataset: A CMVKGDataLoader instance
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        preds_list = []
        targets_list = []
        for sampled_data in tqdm(dataset):
            sampled_data.to(device)
            if hetro:
                y = find_labels_for_batch(batch_data=sampled_data)
                gnn_out = model(sampled_data.x_dict, sampled_data.edge_index_dict)
                out = 0
                for key in gnn_out.keys():
                    if use_max_pooling:
                        gnn_out[key] = global_max_pool(gnn_out[key], sampled_data[key].batch)
                    else:
                        gnn_out[key] = global_mean_pool(gnn_out[key], sampled_data[key].batch)
                    out += gnn_out[key]
                out = F.log_softmax(out, dim=1)
            else:
                y = sampled_data.y
                out = model(sampled_data.x, sampled_data.edge_index, sampled_data.batch)
            preds = torch.argmax(out, dim=1).cpu()
            preds_list.append(preds)
            targets_list.append(y.cpu())
        preds_list = np.concatenate(preds_list)
        targets_list = np.concatenate(targets_list)
        eval_metrics = metrics.compute_metrics(num_labels=constants.NUM_LABELS,
                                               preds=preds_list,
                                               targets=targets_list,
                                               split_name=split_name)
        for metric_name, metric_value in eval_metrics.items():
            wandb.summary[f"eval_{metric_name}"] = metric_value
        return eval_metrics


def create_dataloaders_for_k_fold_cross_validation(graph_dataset: Dataset,
                                                   shuffled_indices,
                                                   batch_size: int,
                                                   val_percent: float,
                                                   test_percent: float,
                                                   k_fold_index: int,
                                                   num_workers: int = 0):
    """

    :param graph_dataset:
    :param shuffled_indices:
    :param batch_size:
    :param val_percent:
    :param test_percent:
    :param k_fold_index:
    :param num_workers:
    :return:
    """
    num_of_examples = len(graph_dataset.dataset)
    test_len = int(test_percent * num_of_examples)
    val_len = int(val_percent * num_of_examples)
    held_out_len = test_len + val_len
    held_out_indices = shuffled_indices[k_fold_index * held_out_len:(k_fold_index + 1) * held_out_len]
    train_indices = shuffled_indices[:k_fold_index * held_out_len]
    train_indices.extend(shuffled_indices[(k_fold_index + 1) * held_out_len:])
    val_indices = held_out_indices[:val_len]
    test_indices = held_out_indices[val_len:]
    dl_train = geom_data.DataLoader(graph_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=SubsetRandomSampler(train_indices))
    dl_val = geom_data.DataLoader(graph_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=SubsetRandomSampler(val_indices))
    dl_test = geom_data.DataLoader(graph_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   sampler=SubsetRandomSampler(test_indices))
    return dl_train, dl_val, dl_test


def create_dataloaders(graph_dataset: Dataset,
                       batch_size: int,
                       val_percent: float,
                       test_percent: float,
                       num_workers: int = 0):
    """Create dataloaders over persuasive argument knowledge graphs.

    :param graph_dataset: A 'CMVKGDataset' instance whose examples correspond to knowledge graphs of persuasive
        arguments.
    :param batch_size: The number of examples processed in a single batch as part of training.
    :param test_percent: The ratio of the original examples dedicated towards a test set. The remaining examples are
        used as part of model training.
    :param num_workers: The number of workers used during training.
    :return:
    """
    num_of_examples = len(graph_dataset.dataset)
    test_len = int(test_percent * num_of_examples)
    val_len = int(val_percent * num_of_examples)
    indexes = random.sample(range(num_of_examples), num_of_examples)
    test_indexes = indexes[:test_len]
    val_indexes = indexes[test_len:test_len + val_len]
    train_indexes = indexes[test_len + val_len:-1]
    dl_train = geom_data.DataLoader(graph_dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=SubsetRandomSampler(train_indexes))
    dl_val = geom_data.DataLoader(graph_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=SubsetRandomSampler(val_indexes))
    dl_test = geom_data.DataLoader(graph_dataset,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   sampler=SubsetRandomSampler(test_indexes))
    return dl_train, dl_val, dl_test


if __name__ == '__main__':
    args = parser.parse_args()
    num_classes = 2
    hetero_type = args.hetero_type
    hetro = args.hetro
    args_dict = vars(args)
    for parameter, value in args_dict.items():
        print(f'{parameter}: {value}')



    num_node_features = constants.BERT_HIDDEN_DIM
    current_path = os.getcwd()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Initializing model: {args.model} device: {device}')
    if args.model == constants.GCN:
        if hetro:
            raise Exception('Hetero GCN is still not implemented')
            model = GCNHetero( num_node_features,
            num_classes=num_classes,
            hidden_layer_dim=args.gcn_hidden_layer_dim,
            use_max_pooling=args.use_max_pooling,
            is_hetro=hetro)
        else:
            model = GCNWithBertEmbeddings(
                num_node_features,
                num_classes=num_classes,
                hidden_layer_dim=args.gcn_hidden_layer_dim,
                use_max_pooling=args.use_max_pooling,
                is_hetro=hetro)
    elif args.model == constants.SAGE:
        model = GraphSage(hidden_channels=args.gcn_hidden_layer_dim,
                          out_channels=num_classes,
                          is_hetro=hetro,
                          use_max_pooling=args.use_max_pooling)
    elif args.model == 'GAT':
        model = GAT(hidden_channels=args.gcn_hidden_layer_dim,
                    out_channels=num_classes,
                    is_hetro=hetro,
                    use_max_pooling=args.use_max_pooling)
    else:
        raise NotImplementedError(f'{args.model} is not implemented')

    # TODO: Creating the knowledge graph datasets takes a lot of time. As of now we make this a one time cost by saving
    #  and loading these datasets. In the future we should optimize the data creation process from a runtime
    #  perspective.
    dir_name = os.path.join(current_path, "cmv_modes", "knowledge_graph_datasets")
    utils.ensure_dir_exists(dir_name)

    if hetro:
        print('Initializing heterophealous dataset')
        if hetero_type == constants.NODES:
            if os.path.exists(os.path.join(dir_name, 'hetro_dataset.pt')):
                kg_dataset = torch.load(os.path.join(dir_name, 'hetro_dataset.pt'))
            else:
                kg_dataset = CMVKGHetroDataset(
                    current_path + "/cmv_modes/change-my-view-modes-master",
                    version=constants.v2_path,
                    debug=args.debug)
                torch.save(kg_dataset, os.path.join(dir_name, 'hetro_dataset.pt'))
        elif hetero_type == constants.EDGES:
            if os.path.exists(os.path.join(dir_name, 'hetro_edges_dataset.pt')):
                kg_dataset = torch.load(os.path.join(dir_name, 'hetro_edges_dataset.pt'))
            else:
                kg_dataset = CMVKGHetroDatasetEdges(
                    current_path + "/cmv_modes/change-my-view-modes-master",
                    version=constants.v2_path,
                    debug=args.debug)
                torch.save(kg_dataset, os.path.join(dir_name, 'hetro_edges_dataset.pt'))
        data = kg_dataset[2]
        print('Converting model to hetero')
        model = to_hetero(model, data.metadata(), aggr='sum')
    else:
        print('initializing homophealous dataset')
        if os.path.exists(os.path.join(dir_name, 'homophelous_dataset.pt')):
            kg_dataset = torch.load(os.path.join(dir_name, 'homophelous_dataset.pt'))
        else:
            kg_dataset = CMVKGDataset(
                current_path + "/cmv_modes/change-my-view-modes-master",
                version=constants.v2_path,
                debug=args.debug)
            torch.save(kg_dataset, os.path.join(dir_name, 'homophelous_dataset.pt'))

    if args.use_k_fold_cross_validation:
        train_metrics = []
        validation_metrics = []
        test_metrics = []
        num_of_examples = len(kg_dataset.dataset)
        shuffled_indices = random.sample(range(num_of_examples), num_of_examples)
        for validation_split_index in range(args.num_cross_validation_splits):
            dl_train, dl_val, dl_test = create_dataloaders_for_k_fold_cross_validation(
                kg_dataset,
                shuffled_indices=shuffled_indices,
                batch_size=args.batch_size,
                val_percent=args.val_percent,
                test_percent=args.test_percent,
                k_fold_index=validation_split_index)
            split_model = copy.deepcopy(model)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay)
            run = wandb.init(project="persuasive_arguments",
                             config=args,
                             name=f"{'heterophelous' if args.hetro else 'homophealous'} "
                                  f"{args.model} with BERT Embeddings and "
                                  f"{'max' if args.use_max_pooling else 'average'} pooling "
                                  f"(split #{validation_split_index + 1})",
                             reinit=True)
            model = train(model=model,
                          training_loader=dl_train,
                          validation_loader=dl_val,
                          epochs=args.num_epochs,
                          optimizer=optimizer,
                          rounds_between_evals=args.rounds_between_evals,
                          hetro=hetro,
                          use_max_pooling=args.use_max_pooling,
                          device=device)
            train_metrics.append(
                eval(model,
                     dl_train,
                     hetro=hetro,
                     use_max_pooling=args.use_max_pooling,
                     device=device,
                     split_name=constants.TRAIN)
            )
            validation_metrics.append(
                eval(model,
                     dl_val,
                     hetro=hetro,
                     use_max_pooling=args.use_max_pooling,
                     device=device,
                     split_name=constants.VALIDATION)
            )
            test_metrics.append(
                eval(model,
                     dl_test,
                     hetro=hetro,
                     use_max_pooling=args.use_max_pooling,
                     device=device,
                     split_name=constants.TEST)
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

    else:
        dl_train, dl_val, dl_test = create_dataloaders(kg_dataset,
                                                       batch_size=args.batch_size,
                                                       val_percent=args.val_percent,
                                                       test_percent=args.test_percent)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        wandb.init(project="persuasive_arguments",
                   config=args,
                   name=f"{'heterophelous' if args.hetro else 'homophealous'} "
                        f"{args.model} with BERT Embeddings and "
                        f"{'max' if args.use_max_pooling else 'average'} pooling")
        model = train(model=model,
                      training_loader=dl_train,
                      validation_loader=dl_val,
                      epochs=args.num_epochs,
                      optimizer=optimizer,
                      rounds_between_evals=args.rounds_between_evals,
                      hetro=hetro,
                      use_max_pooling=args.use_max_pooling,
                      device=device)
        train_metrics = eval(
            model,
            dl_train,
            hetro=hetro,
            use_max_pooling=args.use_max_pooling,
            device=device,
            split_name=constants.TRAIN,
        )
        validation_metrics = eval(
            model,
            dl_val,
            hetro=hetro,
            use_max_pooling=args.use_max_pooling,
            device=device,
            split_name=constants.VALIDATION,
        )
        test_metrics = eval(
            model,
            dl_test,
            hetro=hetro,
            use_max_pooling=args.use_max_pooling,
            device=device,
            split_name=constants.TEST)
        all_metrics = {
            constants.TRAIN: train_metrics,
            constants.VALIDATION: validation_metrics,
            constants.TEST: test_metrics}
        print(all_metrics)
