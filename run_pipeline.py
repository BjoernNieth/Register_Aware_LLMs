from src.models import get_model
from src.prompter import get_prompter
from src.utils.Parser import get_parser
from src.datasets import get_dataset
from src.prompter.Sampler import get_samplers
from src.utils.Random import set_random_seed

from huggingface_hub import login
import os

import json


def main():
    args = get_parser().parse_args()
    login(args.huggingface_token)

    with open(args.config_file_path) as f:
        experiments_config = json.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    for experiment_config in experiments_config:
        output_path = os.path.join(args.output_dir, experiment_config["experiment_name"])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        with open(os.path.join(output_path, "config.json"), mode="w") as f:
            f.write(json.dumps(experiment_config, indent=4))
        # Set randomness seed for libraries to make reproducible experiments
        set_random_seed(int(experiment_config["seed"]))

        # Get the sampler for the LLM 
        samplers = get_samplers(experiment_config["prompts"], int(experiment_config["seed"]))
        for sampler in samplers:
            print(sampler)
        prompter = get_prompter(experiment_config["prompts"]["prompt_type"], samplers)
        print("Testmode is", args.test_mode)
        # Load the datasets and set the n-shot setting
        dataset = get_dataset(experiment_config["dataset"], args.data_path, args.test_mode)
        dataset.set_n_shot(experiment_config["n-shot"])

        # Load the model
        model_config = experiment_config["model"]
        print("init model")
        print("--------------------------------")
        model = get_model(model_config["name"])(model_config["name"],  model_config["model_args"], output_path)

        print("model finished")
        print("--------------------------------")
        # Run inference for the dataset
        prompter.prompt_dataset(dataset, model, experiment_config["prompts"], output_path)
        print("Finished prompting")
        print("--------------------------------")




if __name__ == "__main__":
    main()