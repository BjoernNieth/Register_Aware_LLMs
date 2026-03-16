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


def get_formatted_prompt(prompt, sample):
    """
    Format a prompt without overwriting the original prompt.
    """
    prompt = prompt.copy()
    prompt["content"] = prompt["content"].format(**sample)
    return prompt


class ChatPrompter():
    def __init__(self, samplers):
        self.prompt_samplers = samplers

    def prompt_dataset(self, dataset, model, prompt_configs, output_dir):
        print("start prompting")
        print("--------------------------------")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger = Experiment_Logger_CSV(os.path.join(output_dir, "Query_Log.csv"), len(prompt_configs["iterative_prompts"]))

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
        for iteration in range(math.ceil(len(dataset)/model.batch_size)):
        # Loop over the prompts contained in the config file
            # Some samples might be prompted in multiple iterations
            prompts = [] 
            results = {}
            ids = []
            prompt_log = []
            for i, prompt_config in enumerate(prompt_configs["iterative_prompts"]):
                # Get the databatch
                if i == 0:
                    data_batch = []
                    while len(data_batch) < model.batch_size:
                        try:
                            idx, sample, examples = next(dataset)
                            if result_df is None:
                                data_batch.append((sample, examples))
                                ids.append(idx)
                            else:
                                if idx not in result_df.index:
                                    data_batch.append((sample, examples))
                                    ids.append(idx)
                                else:
                                    print("skipped sample", idx, "already prompted")
                        except StopIteration:
                            break
                        
                for m in range(len(data_batch)):
                    sample, examples = data_batch[m]
                    # Add the user prompt and start of assistant prompt to the prompt
                    if i == 0:
                        if "system_prompt" in prompt_config:
                            prompts.append([prompt_configs["system_prompt"].copy()])
                        else:
                            prompts.append([])
                        if examples is not None:
                            # Write few-shot examples to the prompt
                            prompts[m].extend(write_few_shot_examples(examples, prompt_configs))


                    prompts[m].append(get_formatted_prompt(prompt_config["user_prompt"], sample))
                    prompts[m].append(get_formatted_prompt(prompt_config["assistant_prompt_start"], sample))

                if len(prompts) != 0:
                    outputs = model.generate(prompts, self.prompt_samplers[i])
                    
                    for m in range(len(outputs)):
                        output = clean_string(outputs[m])
                        idx, sample, examples = data_batch[m]
                        if "output_length" in prompt_config:
                            output_length = int(prompt_config["output_length"].format(**sample))
                            if len(output) > output_length:
                                output = output[:output_length]

                        # Check if output of model should be parsed
                        if prompt_config["parse_output"] is True:
                            # Check if the key is already in the results dictionary 
                            if prompt_config["output_name"] not in results:
                                results[prompt_config["output_name"]] = []

                            results[prompt_config["output_name"]].append(output)

                        # If in first iteration, add the id to prompt log, else just the prompt and output
                        if i == 0:
                            prompt_log.append([idx, prompts[m], output])
                        else:
                            prompt_log[m].extend([prompts[m], output])

                        # Check if one more round if iterative prompting is performed
                        if i + 1 < len(prompt_configs["iterative_prompts"]):
                            # Append the output to the last prompt in the list for the next iteration
                            prompts[m][-1]["content"] += output

            # Save the results to disk 
            if result_df is None:
                result_df = pd.DataFrame(results, index=ids)
                result_df.to_csv(output_csv_path, header=True)
            else:
                pd.DataFrame(results, index=ids).to_csv(output_csv_path, mode="a", header=False)
                result_df = pd.concat([result_df, pd.DataFrame(results, index=ids)])
            
            for line in prompt_log:
                logger.log_prompt(line)

        
        return result_df
