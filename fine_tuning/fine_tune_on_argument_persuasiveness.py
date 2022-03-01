import argparse
import logging
import os
import wandb
import sys
import transformers

# This is necessary when running this script from the server.
try:
    import constants
    import fine_tuning
    import preprocessing
except ModuleNotFoundError as e:
    # Extend the persuasive_argumentation package.
    package_path = os.sep.join(os.getcwd().split(os.sep)[:-1])
    print(f'Adding package path {package_path} to "sys.path"')
    sys.path.extend([package_path])
    import constants
    import fine_tuning
    import preprocessing

"""
Example command on Newton Cluster:
srun --gres=gpu:1 -p nlp python3 fine_tune_on_argument_persuasiveness.py \
    --fine_tuning_num_training_epochs 4 \
    --fine_tuning_learning_rate 1e-5 \
    --fine_tuning_per_device_train_batch_size 8 \
    --fine_tuning_per_device_eval_batch_size 16
"""


parser = argparse.ArgumentParser(
    description='Process flags for fine-tuning transformers on an argumentation downstream task.')
parser.add_argument('--fine_tuning_dataset_name',
                    type=str,
                    default=constants.CMV_DATASET_NAME,
                    required=False,
                    help='The name of the file in which the downstream dataset is stored.')
parser.add_argument('--fine_tuning_model_checkpoint_name',
                    type=str,
                    default=constants.BERT_BASE_CASED,
                    required=False,
                    help="The name of the checkpoint from which we load our model and tokenizer.")
parser.add_argument('--fine_tuning_num_training_epochs',
                    type=int,
                    default=4,
                    required=False,
                    help="The number of training rounds over the dataset.")
parser.add_argument('--fine_tuning_learning_rate',
                    type=float,
                    default=1e-3,
                    required=False,
                    help="The learning rate used by the fine-tuning model for the downstream task.")
parser.add_argument('--fine_tuning_output_dir',
                    type=str,
                    default='./results',
                    required=False,
                    help="The directory in which model results are stored.")
parser.add_argument('--fine_tuning_logging_dir',
                    type=str,
                    default="./logs",
                    required=False,
                    help="The directory in which the model stores logs.")
parser.add_argument('--fine_tuning_per_device_train_batch_size',
                    type=int,
                    default=16,
                    help="The number of examples per batch per device during training.")
parser.add_argument('--fine_tuning_per_device_eval_batch_size',
                    type=int,
                    default=64,
                    help="The number of examples per batch per device during evaluation.")
parser.add_argument('--fine_tuning_warmup_steps',
                    type=int,
                    default=500,
                    help="The number of warmup steps the model takes at the start of training.")
parser.add_argument('--eval_steps',
                    type=int,
                    default=500,
                    help="Perform evaluation every 'eval_steps' steps.")
parser.add_argument('--fine_tuning_weight_decay',
                    type=float,
                    default=0.01,
                    help="The weight decay parameter supplied to the optimizer for use during training.")
parser.add_argument('--fine_tuning_logging_steps',
                    type=int,
                    default=10,
                    help="The number of steps a model takes between recording to logs.")
parser.add_argument('--fine_tuning_wandb_entity',
                    type=str,
                    default='noambenmoshe',
                    help="The wandb entity used to track training.")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    args = parser.parse_args()

    if args.fine_tuning_wandb_entity:
        wandb.init(project="persuasive_arguments", entity=args.fine_tuning_wandb_entity)

    args_dict = vars(args)
    for parameter, value in args_dict.items():
        print(f'{parameter}: {value}')

    configuration = transformers.TrainingArguments(
        output_dir=args.fine_tuning_output_dir,
        num_train_epochs=args.fine_tuning_num_training_epochs,
        per_device_train_batch_size=args.fine_tuning_per_device_train_batch_size,
        per_device_eval_batch_size=args.fine_tuning_per_device_eval_batch_size,
        eval_steps=args.eval_steps,
        evaluation_strategy=transformers.training_args.IntervalStrategy('steps'),
        learning_rate=args.fine_tuning_learning_rate,
        warmup_steps=args.fine_tuning_warmup_steps,
        weight_decay=args.fine_tuning_weight_decay,
        logging_dir=args.fine_tuning_logging_dir,
        logging_steps=args.fine_tuning_logging_steps,
        report_to=["wandb"],
    )
    model = transformers.BertForSequenceClassification.from_pretrained(
        args.fine_tuning_model_checkpoint_name,
        num_labels=constants.NUM_LABELS)
    dataset = (
        preprocessing.get_dataset(task_name=constants.BINARY_CMV_DELTA_PREDICTION,
                                  tokenizer=transformers.BertTokenizer.from_pretrained(constants.BERT_BASE_CASED),
                                  save_text_datasets=True,
                                  dataset_name=args.fine_tuning_dataset_name))
    fine_tuned_model, downstream_metrics = (
        fine_tuning.fine_tune_on_task(dataset=dataset,
                                      model=model,
                                      configuration=configuration,
                                      task_name=constants.BINARY_CMV_DELTA_PREDICTION,
                                      is_probing=False,
                                      logger=logger))
