import fine_tuning
import constants

import transformers
import argparse

import wandb

"""
Example command on Newton Cluster:
srun --gres=gpu:1 -p nlp python3 fine_tune_on_argument_persuasiveness.py \
    --fine_tuning_num_training_epochs 10 \
    --fine_tuning_per_device_train_batch_size 16 \
    --fine_tuning_per_device_eval_batch_size 32
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
                    default='zbamberger',
                    help="The wandb entity used to track training.")

if __name__ == "__main__":
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
        warmup_steps=args.fine_tuning_warmup_steps,
        weight_decay=args.fine_tuning_weight_decay,
        logging_dir=args.fine_tuning_logging_dir,
        logging_steps=args.fine_tuning_logging_steps,
        report_to=["wandb"],
    )
    model = transformers.BertForSequenceClassification.from_pretrained(
        args.fine_tuning_model_checkpoint_name,
        num_labels=constants.NUM_LABELS)

    fine_tuned_trainer, downstream_metrics = fine_tuning.fine_tune_on_downstream_task(
        dataset_name=args.fine_tuning_dataset_name,
        model=model,
        configuration=configuration)
