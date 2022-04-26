import wandb

# Experiment parameters.
# Each unique combination of parameters from the below lists constitutes a unique experiment.
graph_types = ["homophilous", "heterophilous"]
graph_architectures = ["gcn", "gat", "sage"]
pooling_types = ["max", "avg"]

if __name__ == "__main__":
    generated_sweep_ids = {}
    for graph_type in graph_types:
        for graph_architecture in graph_architectures:
            for pooling_type in pooling_types:
                sweep_config = ({
                    "name": f"{graph_type} {graph_architecture} ({pooling_type} pooling)",
                    "method": "grid",
                    "metric": {
                      "name": "eval_test_accuracy",
                      "goal": "maximize",
                    },
                    "parameters": {
                        # Experiment defining hyper-parameters.
                        "hetro": {
                            "value": "True" if graph_architecture == "heterophilous" else ""
                        },
                        "model": {
                            "value": graph_architecture
                        },
                        # Fixed hyper-parameters.
                        "num_epochs": {
                            "value": 30
                        },
                        "batch_size": {
                            "value": 16,
                        },
                        "test_percent": {
                            "value": 0.1,
                        },
                        "val_percent": {
                            "value": 0.1,
                        },
                        "rounds_between_evals": {
                            "value": 1,
                        },
                        "debug": {
                            "value": "",
                        },
                        # Searched hyper-parameters
                        "learning_rate": {
                            "values": [1e-4, 1e-3, 1e-2]
                        },
                        "seed": {
                            "values": [1, 2, 3]
                        },
                        "gcn_hidden_layer_dim": {
                            "values": [128, 256, 512]
                        },
                        "scheduler_gamma": {
                            "values": [0.9, 0.99]
                        }
                    },
                })
                generated_sweep_ids[sweep_config["name"]] = wandb.sweep(
                    sweep_config,
                    project="persuasive_argumentation",
                    entity="persuasive_arguments"
                )
    for sweep_name, sweep_id in generated_sweep_ids.items():
        print(f"{sweep_name}:\n"
              f"\tsrun --gres=gpu:1 -p nlp wandb agent persuasive_arguments/persuasive_argumentation/{sweep_id}")
