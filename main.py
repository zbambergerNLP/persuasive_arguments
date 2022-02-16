from __future__ import annotations

import datasets
import torch

import constants
import fine_tuning.fine_tuning as fine_tuning
import preprocessing
import probing.probing as probing

import argparse
import logging
import os
import transformers
import wandb

"""
How to run fine-tuning on the relation extraction probing task:
srun --gres=gpu:1 -p nlp python3 main.py \
    --fine_tune_model_on_argument_relations True \
    --probing_num_training_epochs 8
    
How to run probing experiments:
srun --gres=gpu:1 -p nlp python3 main.py --probing_model "logistic_regression" \
    --fine_tuned_model_path "results/checkpoint-500" \
    --fine_tune_model_on_argument_relations True \
    --probing_num_training_epochs 8
"""

parser = argparse.ArgumentParser(
    description='Process flags for fine-tuning transformers on an argumentation downstream task.')
parser.add_argument('--probing_wandb_entity',
                    type=str,
                    default='zbamberger',
                    help="The wandb entity used to track training.")


# General probing parameters
parser.add_argument('--probing_model',
                    type=str,
                    default=constants.MLP,
                    required=False,
                    help="The string name of the model type used for probing. Either logistic regression or MLP.")
parser.add_argument('--fine_tuned_model_path',
                    type=str,
                    required=False,
                    default=os.path.join(constants.RESULTS, 'checkpoint-1500'),
                    help='The path fine-tuned model trained on the argument persuasiveness prediction task.')
parser.add_argument('--model_checkpoint_name',
                    type=str,
                    default=constants.BERT_BASE_CASED,
                    required=False,
                    help="The name of the checkpoint from which we load our model and tokenizer.")
parser.add_argument('--probing_model_learning_rate',
                    type=float,
                    default=5e-3,
                    required=False,
                    help="The learning rate used by the probing model for the probing task.")
parser.add_argument('--fine_tuning_on_probing_task_learning_rate',
                    type=float,
                    default=1e-5,
                    required=False,
                    help="The learning rate used for fine tuning a transformer on the probing task.")
parser.add_argument('--probing_model_scheduler_gamma',
                    type=float,
                    default=0.9,
                    help="Decays the learning rate of each parameter group by gamma every epoch.")


# Probing on intra-argument relations:
parser.add_argument('--probe_model_on_intra_argument_relations',
                    type=bool,
                    default=True,
                    required=False,
                    help="Whether or not a pre-trained transformer language model should be trained and evaluated on a"
                         "probing task. In this case, the probing task involves classifying whether or not a relation"
                         "exist between the first argument preposition (either a claim or premise), and a second "
                         "argument preposition.")
parser.add_argument('--generate_new_relations_probing_dataset',
                    type=bool,
                    default=False,
                    required=False,
                    help='True instructs the fine-tuned model to generate new hidden embeddings corresponding to each'
                         ' example. These embeddings serve as the input to an MLP probing model. False assumes that'
                         ' such a dataset already exists, and is stored in json file within the ./probing directory.')
parser.add_argument('--fine_tune_model_on_argument_relations',
                    type=bool,
                    default=False,
                    required=False,
                    help='Fine tune the model specified in `model_checkpoint_name` on the relation prediction probing'
                         ' dataset.')


# Probing on premise modes:
parser.add_argument('--probe_model_on_premise_modes',
                    type=bool,
                    default=True,
                    required=False,
                    help=('Whether or not a pre-trained transformer language model should be trained and evaluated'
                          'on a probing task. In this case, the probing task involves classifying the argumentation '
                          'mode (i.e., the presence of ethos, logos, or pathos) within a premise.'))
parser.add_argument('--premise_modes',
                    type=set,
                    default={constants.LOGOS, constants.PATHOS},
                    required=False,
                    help="The premise modes we use for our probing experiments on binary premise mode prediction.")
parser.add_argument('--generate_new_premise_mode_probing_dataset',
                    type=bool,
                    default=False,
                    required=False,
                    help='True instructs the fine-tuned model to generate new hidden embeddings corresponding to each'
                         ' example. These embeddings serve as the input to an MLP probing model. False assumes that'
                         ' such a dataset already exists, and is stored in json file within the ./probing directory.')
parser.add_argument('--fine_tune_model_on_premise_modes',
                    type=bool,
                    default=False,
                    required=False,
                    help='Fine tune the model specified in `model_checkpoint_name` on the probing datasets.')
parser.add_argument('--probe_model_fine_tuned_on_probing_task',
                    type=bool,
                    default=False,
                    required=False,
                    help='True if we wish to perform probing on a model fine-tuned on the probing task. False '
                         'otherwise. This "debug" setting is meant to test the validity of the probing infrastructure.')
parser.add_argument('--multi_class_premise_mode_probing',
                    type=bool,
                    default=True,
                    required=False,
                    help='True if the label space for classifying premise mode (probing task) is the superset of '
                         '{"ethos", "logos", "pathos}. False if the label space is binary, where True indicates that'
                         'the premise entails the dataset-specific argumentative mode.')


# Configuration flags:
parser.add_argument('--probing_output_dir',
                    type=str,
                    default='./results',
                    required=False,
                    help="The directory in which probing model results are stored.")
parser.add_argument('--fine_tuning_on_probing_task_num_training_epochs',
                    type=int,
                    default=4,
                    required=False,
                    help="The number of training rounds for fine-tuning on the probing dataset.")
parser.add_argument('--probing_num_training_epochs',
                    type=int,
                    default=40,
                    required=False,
                    help="The number of training rounds over the probing dataset.")
parser.add_argument('--probing_per_device_train_batch_size',
                    type=int,
                    default=16,
                    help="The number of examples per batch per device during probe training.")
parser.add_argument('--probing_per_device_eval_batch_size',
                    type=int,
                    default=64,
                    help="The number of examples per batch per device during probe evaluation.")
parser.add_argument('--eval_steps',
                    type=int,
                    default=50,
                    help="Perform evaluation every 'eval_steps' steps.")
parser.add_argument('--probing_warmup_steps',
                    type=int,
                    default=200,
                    help="The number of warmup steps the model takes at the start of probing.")
parser.add_argument('--probing_weight_decay',
                    type=float,
                    default=0.01,
                    help="The weight decay parameter supplied to the optimizer for use during probing.")
parser.add_argument('--probing_logging_dir',
                    type=str,
                    default="./logs",
                    required=False,
                    help="The directory in which the model stores logs.")
parser.add_argument('--probing_logging_steps',
                    type=int,
                    default=10,
                    help="The number of steps a model takes between recording to logs.")


def run_fine_tuning(probing_wandb_entity: str,
                    task_name: str,
                    dataset: datasets.Dataset,
                    model: transformers.PreTrainedModel | torch.torch.nn.Module,
                    configuration: transformers.TrainingArguments,
                    is_probing: bool,
                    premise_mode: str = None,
                    logger: logging.Logger = None):
    """

    :param probing_wandb_entity: The wandb entity used to track training.
    :param task_name: A string. One of {'multiclass', 'binary_premise_mode_prediction', 'intra_argument_relations',
        'binary_cmv_delta_prediction'}.
    :param dataset: The dataset on which we fine-tune the given model.
    :param model: A pretrained transformer language model for sequence classification.
    :param configuration: A 'transformers.TrainingArguments' instance.
    :param is_probing: True if the task on which we fine tune the model is a probing task. False if the task is a
        downstream task.
    :param premise_mode: A string representing the premise mode towards which the dataset is oriented. For example,
        if the mode were 'ethos', then positive labels would be premises who's label contains 'ethos'.
    :param logger: A logging.Logger instance used for logging.
    :return:  A 2-tuple of the form (trainer, eval_metrics). The trainer is a 'transformers.Trainer' instance used to
        fine-tune the model, and the metrics are a dictionary derived from evaluating the model on the verification set.
    """
    run_name = f'Fine tune on {task_name}'
    if premise_mode:
        run_name += f' ({premise_mode})'
    if probing_wandb_entity:
        run = wandb.init(project="persuasive_arguments",
                         entity=probing_wandb_entity,
                         reinit=True,
                         name=run_name)
    trainer, eval_metrics = (
        fine_tuning.fine_tune_on_task(dataset=dataset,
                                      model=model,
                                      configuration=configuration,
                                      task_name=task_name,
                                      is_probing=is_probing,
                                      premise_mode=premise_mode,
                                      logger=logger))
    prefix = f'{task_name} ({premise_mode})' if premise_mode else f'{task_name}'
    print(f'{prefix} eval metrics:\n{eval_metrics}')
    if args.probing_wandb_entity:
        run.finish()
    return trainer, eval_metrics


if __name__ == "__main__":

    # TODO: Create a function to extract the most recent checkpoint in a directory. Leverage that function when loading
    #  a fine-tuned model.
    logger = logging.getLogger(__name__)
    args = parser.parse_args()

    args_dict = vars(args)
    for parameter, value in args_dict.items():
        print(f'{parameter}: {value}')

    configuration = transformers.TrainingArguments(
        output_dir=args.probing_output_dir,
        num_train_epochs=args.fine_tuning_on_probing_task_num_training_epochs,
        eval_steps=args.eval_steps,
        evaluation_strategy='steps',
        learning_rate=args.fine_tuning_on_probing_task_learning_rate,
        per_device_train_batch_size=args.probing_per_device_train_batch_size,
        per_device_eval_batch_size=args.probing_per_device_eval_batch_size,
        warmup_steps=args.probing_warmup_steps,
        weight_decay=args.probing_weight_decay,
        logging_dir=args.probing_logging_dir,
        logging_steps=args.probing_logging_steps,
        report_to=["wandb"],
    )
    model = transformers.BertForSequenceClassification.from_pretrained(
        args.model_checkpoint_name,
        num_labels=constants.NUM_LABELS)

    # We are now considering model performance on argumentative probing tasks.
    current_path = os.getcwd()
    probing_dir_path = os.path.join(current_path, constants.PROBING)
    tokenizer = transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED)

    if args.probe_model_on_intra_argument_relations:
        intra_argument_relations_probing_dataset = preprocessing.get_dataset(
            task_name=constants.INTRA_ARGUMENT_RELATIONS,
            tokenizer=tokenizer)
        if args.fine_tune_model_on_argument_relations:
            intra_argument_relations_trainer, intra_argument_relations_eval_metrics = (
                run_fine_tuning(probing_wandb_entity=args.probing_wandb_entity,
                                task_name=constants.INTRA_ARGUMENT_RELATIONS,
                                dataset=intra_argument_relations_probing_dataset,
                                model=model,
                                configuration=configuration,
                                is_probing=True,
                                logger=logger))
        if args.probe_model_on_intra_argument_relations:
            run = wandb.init(project="persuasive_arguments",
                             entity=args.probing_wandb_entity,
                             reinit=True,
                             name='Intra argument relations probing')
            pretrained_probing_model, fine_tuned_probing_model, eval_metrics = probing.probe_model_on_task(
                probing_dataset=intra_argument_relations_probing_dataset,
                probing_model=args.probing_model,
                generate_new_hidden_state_dataset=args.generate_new_relations_probing_dataset,
                task_name=constants.INTRA_ARGUMENT_RELATIONS,
                pretrained_checkpoint_name=args.model_checkpoint_name,
                fine_tuned_model_path=args.fine_tuned_model_path,
                mlp_learning_rate=args.probing_model_learning_rate,
                mlp_training_batch_size=args.probing_per_device_train_batch_size,
                mlp_eval_batch_size=args.probing_per_device_eval_batch_size,
                mlp_num_epochs=args.probing_num_training_epochs,
                mlp_optimizer_scheduler_gamma=args.probing_model_scheduler_gamma)
            run.finish()
            print('pretrained probe results:')
            print(eval_metrics[constants.PRETRAINED][constants.CONFUSION_MATRIX])
            print(eval_metrics[constants.PRETRAINED][constants.CLASSIFICATION_REPORT])
            print('fine tuned probe results:')
            print(eval_metrics[constants.FINE_TUNED][constants.CONFUSION_MATRIX])
            print(eval_metrics[constants.FINE_TUNED][constants.CLASSIFICATION_REPORT])

    if args.multi_class_premise_mode_probing:
        pretrained_multiclass_model = transformers.BertForSequenceClassification.from_pretrained(
            args.model_checkpoint_name,
            num_labels=len(constants.PREMISE_MODE_TO_INT))
        multi_class_premise_mode_dataset = preprocessing.get_dataset(task_name=constants.MULTICLASS,
                                                                     tokenizer=tokenizer)
        if args.fine_tune_model_on_premise_modes:
            multi_class_tainer, multi_class_eval_metrics = (
                run_fine_tuning(probing_wandb_entity=args.probing_wandb_entity,
                                task_name=constants.MULTICLASS,
                                dataset=multi_class_premise_mode_dataset,
                                model=pretrained_multiclass_model,
                                configuration=configuration,
                                is_probing=True,
                                logger=logger))
        if args.probe_model_on_premise_modes:
            run = wandb.init(project="persuasive_arguments",
                             entity=args.probing_wandb_entity,
                             reinit=True,
                             name='Multiclass premise model probing')
            pretrained_probing_model, _, eval_metrics = probing.probe_model_on_task(
                probing_dataset=multi_class_premise_mode_dataset,
                probing_model=args.probing_model,
                generate_new_hidden_state_dataset=args.generate_new_premise_mode_probing_dataset,
                task_name=constants.MULTICLASS,
                pretrained_checkpoint_name=args.model_checkpoint_name,
                mlp_learning_rate=args.probing_model_learning_rate,
                mlp_training_batch_size=args.probing_per_device_train_batch_size,
                mlp_eval_batch_size=args.probing_per_device_eval_batch_size,
                mlp_num_epochs=args.probing_num_training_epochs,
                mlp_optimizer_scheduler_gamma=args.probing_model_scheduler_gamma)
            run.finish()
            print('multiclass probe results:')
            print(eval_metrics[constants.CONFUSION_MATRIX])
            print(eval_metrics[constants.CLASSIFICATION_REPORT])

    logos_dataset = preprocessing.get_dataset(task_name=constants.BINARY_PREMISE_MODE_PREDICTION,
                                              tokenizer=tokenizer,
                                              premise_mode=constants.LOGOS)
    pathos_dataset = preprocessing.get_dataset(task_name=constants.BINARY_PREMISE_MODE_PREDICTION,
                                               tokenizer=tokenizer,
                                               premise_mode=constants.PATHOS)

    premise_modes_dataset_dict = {constants.LOGOS: logos_dataset,
                                  constants.PATHOS: pathos_dataset}

    # Perform fine-tuning on each of the premise mode binary classification tasks.
    if args.fine_tune_model_on_premise_modes:
        for premise_mode in args.premise_modes:
            dataset = premise_modes_dataset_dict[premise_mode]
            run_fine_tuning(probing_wandb_entity=args.probing_wandb_entity,
                            task_name=constants.BINARY_PREMISE_MODE_PREDICTION,
                            premise_mode=premise_mode,
                            dataset=dataset,
                            model=model,
                            configuration=configuration,
                            is_probing=True,
                            logger=logger)

    # Perform probing on each of the premise mode binary classification tasks.
    if args.probe_model_on_premise_modes:
        for premise_mode in args.premise_modes:
            dataset = premise_modes_dataset_dict[premise_mode]
            run = wandb.init(project="persuasive_arguments",
                             entity=args.probing_wandb_entity,
                             reinit=True,
                             name=f'Binary premise mode prediction ({premise_mode}) probing')
            retrained_probing_model, fine_tuned_probing_model, eval_metrics = (
                probing.probe_model_on_task(
                    probing_dataset=dataset,
                    probing_model=args.probing_model,
                    generate_new_hidden_state_dataset=args.generate_new_premise_mode_probing_dataset,
                    task_name=constants.BINARY_PREMISE_MODE_PREDICTION,
                    pretrained_checkpoint_name=args.model_checkpoint_name,
                    fine_tuned_model_path=args.fine_tuned_model_path,
                    mlp_learning_rate=args.probing_model_learning_rate,
                    mlp_training_batch_size=args.probing_per_device_train_batch_size,
                    mlp_eval_batch_size=args.probing_per_device_eval_batch_size,
                    mlp_num_epochs=args.probing_num_training_epochs,
                    mlp_optimizer_scheduler_gamma=args.probing_model_scheduler_gamma,
                    premise_mode=premise_mode))
            run.finish()
            print(f'{constants.BINARY_PREMISE_MODE_PREDICTION} ({premise_mode}) pretrained probe results:')
            print(eval_metrics[constants.PRETRAINED][constants.CONFUSION_MATRIX])
            print(eval_metrics[constants.PRETRAINED][constants.CLASSIFICATION_REPORT])
            print(f'{constants.BINARY_PREMISE_MODE_PREDICTION} ({premise_mode}) fine tuned probe results:')
            print(eval_metrics[constants.FINE_TUNED][constants.CONFUSION_MATRIX])
            print(eval_metrics[constants.FINE_TUNED][constants.CLASSIFICATION_REPORT])
