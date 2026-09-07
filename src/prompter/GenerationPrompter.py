import json
import os
import pandas as pd
from ..utils.Experiment_Data_Logger import Experiment_Logger_CSV
import re
import math

def clean_string(s):
    # Replace line breaks and tabs with a space
    s = s.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    # Collapse multiple spaces into one
    s = re.sub(r'\s+', ' ', s)
    # Remove leading and trailing spaces
    return s.strip()

def write_few_shot_examples(examples, prompt_configs):
    """
    Write n few-shot examples to the start of the prompt
    """
    example_prompts = []
    for example in examples:
        for prompt_config in prompt_configs:
            example_user_prompt = prompt_config["user_prompt"]
            example_user_prompt["content"] = example_user_prompt["content"].format(**example)
            example_prompts.append(example_user_prompt)

            example_system_answer = prompt_config["assistant_prompt_few_shot"]
            example_system_answer["content"] = example_system_answer["content"].format(**example)
            example_prompts.append(example_system_answer)

    return example_prompts  



class GenerationPrompter():
    def __init__(self, samplers):
        self.prompt_samplers = samplers

    def prompt_dataset(self, dataset, model, prompt_config, output_dir):
        print("start prompting")
        print("--------------------------------")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger = Experiment_Logger_CSV(os.path.join(output_dir, "Query_Log.csv"), 1)

        # Build output directory of experiment
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check if there are already experiment results
        output_csv_path = os.path.join(output_dir, "Model_Output.csv")
        if os.path.isfile(output_csv_path):
            result_df = pd.read_csv(output_csv_path, index_col=0)
        else:
            result_df = None
        
        iter(dataset)
        for _ in range(math.ceil(len(dataset)/model.batch_size)):
        # Loop over the prompts contained in the config file
            # Some samples might be prompted in multiple iterations
            prompts = [] 

            data_batch = []
            while len(data_batch) < model.batch_size:
                try:
                    idx, sample, examples = next(dataset)
                    if result_df is None:
                        data_batch.append((idx, sample, examples))
                    else:
                        if idx not in result_df.index:
                            data_batch.append((idx, sample, examples))
                        else:
                            print("skipped sample", idx, "already prompted")
                except StopIteration:
                    break
                        
            for m in range(len(data_batch)):
                idx, sample, examples = data_batch[m]
                prompts.append(prompt_config["prompt"].format(**sample))


            if len(prompts) != 0:
                print(prompts)
                outputs = model.generate(prompts, self.prompt_samplers[0])
                #[0].outputs[0].text
                for m in range(len(outputs)):
                    output = clean_string(outputs[m])
                    idx, sample, examples = data_batch[m]


                    result = {}
                    result[prompt_config["output_name"]] = [output]
                        # Save results of experiment to disk
                    if result_df is None:
                        result_df = pd.DataFrame(result, index=[idx])
                        result_df.to_csv(output_csv_path, header=True)
                    else:
                        pd.DataFrame(result, index=[idx]).to_csv(output_csv_path, mode="a", header=False)
                        result_df = pd.concat([result_df, pd.DataFrame(result, index=[idx])])
                    
                    logger.log_prompt([idx, prompts[m], output])
        
        return result_df


