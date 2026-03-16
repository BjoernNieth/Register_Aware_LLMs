import json
import os
import pandas as pd
from ..utils.Experiment_Data_Logger import Experiment_Logger_CSV
import re
import math
import dill
import time
from queue import LifoQueue
import unicodedata

def flatten_string(s):
    if not isinstance(s, str):
        return s

    # Normalize Unicode
    s = unicodedata.normalize("NFKC", s)

    # Replace all problematic Unicode line/space characters
    s = s.replace('\u2028', ' ')  # Line separator
    s = s.replace('\u2029', ' ')  # Paragraph separator
    s = s.replace('\u00A0', ' ')  # Non-breaking space
    s = s.replace('\u200B', '')   # Zero-width space
    s = s.replace('\r', ' ')
    s = s.replace('\n', ' ')
    s = s.replace('\t', ' ')

    # Collapse all whitespace to single space
    s = re.sub(r'\s+', ' ', s)

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


class BachtProcessingChatPrompter():
    """Prompter for batch processing of chat prompts. This prompter must be called multiple times to eventually prompt the whole conversation"""
    def __init__(self, samplers):
        self.prompt_samplers = samplers

    def prompt_dataset(self, dataset, model, prompt_configs, output_dir):
        print("start prompting")
        print(f"Dataset {dataset.name} consists of {len(dataset)} samples")
        print("--------------------------------")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        error_counter = 0
        columns = []
        for prompt_config in prompt_configs["iterative_prompts"]:
            if prompt_config["parse_output"] is True:
                columns.append(prompt_config["output_name"])

        if os.path.isfile(os.path.join(output_dir, "Model_Output.csv")):
            result_df = pd.read_csv(os.path.join(output_dir, "Model_Output.csv"), index_col=0)
        else:
            result_df = pd.DataFrame(columns=columns)


        
        if os.path.isfile(os.path.join(output_dir, "Query_Log.csv")):
            query_log_df = pd.read_csv(os.path.join(output_dir, "Query_Log.csv"), index_col=0)
            query_log_columns = list(query_log_df.columns)
        else:
            query_log_columns = []
            for i in range(len(prompt_configs["iterative_prompts"])):
                query_log_columns.append(f"query_{i}")
                query_log_columns.append(f"response_{i}")
            query_log_df = pd.DataFrame(columns=query_log_columns)
        
        queue_path = os.path.join(output_dir, "prompt_queue.dill")
        if os.path.isfile(queue_path):
            print("Resuming from previous queue.")
            new_queue = False
            with open(queue_path, "rb") as f:
                prompt_queue = dill.load(f)
            if prompt_queue.empty():
                print("Queue is empty, starting a new queue.")
                new_queue = True
        else:
            new_queue = True
            prompt_queue = LifoQueue(len(dataset))

        last_batch_number = 0
        element = {
            "ids": [],
            "iteration": 0,
            # queue | submitted  
            "status": "queued",
            "prompts": [],
            "samples": [],
            "examples": [],
            "batch_job": None,
            "batch_number": last_batch_number
        }
        for idx, sample, examples in dataset:
            if new_queue:
                element["ids"].append(idx)
                element["samples"].append(sample)
                element["examples"].append(examples)
                element["prompts"].append([])
                element["iteration"] = 0
                if len(element["ids"]) >= 100:
                    prompt_queue.put(element)
                    element = {
                        "ids": [],
                        "iteration": 0,
                        # queue | submitted  
                        "status": "queued",
                        "prompts": [],
                        "samples": [],
                        "examples": [],
                        "batch_job": None,
                        "batch_number": last_batch_number + 1
                    }
                    last_batch_number += 1
            if idx not in result_df.index:
                result_df.loc[idx] = [None] * len(columns)
            if idx not in query_log_df.index:
                query_log_df.loc[idx] = [None] * len(query_log_columns)
        
        if new_queue and len(element["ids"]) > 0:
            prompt_queue.put(element)
        
        print(f"Queue size: {prompt_queue.qsize()}")

        # Loop over the queue until it is empty
        while not prompt_queue.empty():
            # Get the next element from the FIFO queue
            element = prompt_queue.get()
            already_prompted_positions = []
            already_prompted_element = {
                "ids": [],
                "iteration": element["iteration"] + 1,
                # queue | submitted  
                "status": "queued",
                "prompts": [],
                "samples": [],
                "examples": [],
                "batch_job": None,
                "batch_number": last_batch_number
            }
            
            if element["status"] == "queued" and element["iteration"] < len(prompt_configs["iterative_prompts"]):
                batch_name = f"{dataset.name}_batch_{element['batch_number']}_{element['iteration']}"
                # If in the first iteration, check for system prompt and few-shot examples
                for m in range(len(element["prompts"])):
                    if element["iteration"] == 0:
                        sample = element["samples"][m]

                        if "system_prompt" in prompt_configs:
                            element["prompts"][m].append(prompt_configs["system_prompt"].copy())

                        if element["examples"][m] is not None:
                            # Write few-shot examples to the prompt
                            element["prompts"][m].extend(write_few_shot_examples(element["examples"][m], prompt_configs))

                    # Add the user prompt and start of assistant prompt to the prompt
                    element["prompts"][m].append(get_formatted_prompt(prompt_configs["iterative_prompts"][element["iteration"]]["user_prompt"], sample))
                    element["prompts"][m].append(get_formatted_prompt(prompt_configs["iterative_prompts"][element["iteration"]]["assistant_prompt_start"], sample))

                    if pd.notnull(result_df.loc[element["ids"][m], prompt_configs["iterative_prompts"][element["iteration"]]["output_name"]]):
                        print(f"skipped sample {element['ids'][m]} already prompted")
                        already_prompted_positions.append(m)
                        if element["iteration"] + 1 < len(prompt_configs["iterative_prompts"]):
                            element["prompts"][m][-1]["content"] += result_df.loc[element["ids"][m], prompt_configs["iterative_prompts"][element["iteration"]]["output_name"]]

                for m in already_prompted_positions:
                    already_prompted_element["ids"].append(element["ids"][m])
                    already_prompted_element["prompts"].append(element["prompts"][m])
                    already_prompted_element["samples"].append(element["samples"][m])
                    already_prompted_element["examples"].append(element["examples"][m])
                    

                element["ids"] = [element["ids"][m] for m in range(len(element["ids"])) if m not in already_prompted_positions]
                element["prompts"] = [element["prompts"][m] for m in range(len(element["prompts"])) if m not in already_prompted_positions]
                element["samples"] = [element["samples"][m] for m in range(len(element["samples"])) if m not in already_prompted_positions]
                element["examples"] = [element["examples"][m] for m in range(len(element["examples"])) if m not in already_prompted_positions] 

                if len(already_prompted_element["ids"]) > 0:
                    print(f"Reinserting {len(already_prompted_element['ids'])} already prompted samples into the queue.")
                    prompt_queue.put(already_prompted_element)
                    last_batch_number += 1
                    with open(queue_path, "wb") as dill_file:
                        dill.dump(prompt_queue, dill_file)

                if len(element["ids"]) > 0:
                    print(f"{batch_name} is the first time submitting these samples: {len(element['ids'])}.")
                    model.write_batch(element["prompts"], self.prompt_samplers[element["iteration"]], batch_name)
                    batch_job = model.query_batch(batch_name)
                    
                    
                    time.sleep(1)
                    element["batch_job"] = batch_job
                    element["status"] = "submitted"
                    
                    # Reinsert the element into the queue for further processing
                    prompt_queue.put(element)
                    with open(queue_path, "wb") as dill_file:
                        dill.dump(prompt_queue, dill_file)  

            # Check if the batch job is completed
            elif element["status"] == "submitted":
                batch_name = f"{dataset.name}_batch_{element['batch_number']}_{element['iteration']}"
                try:
                    is_ready = model.batch_completed(element["batch_job"])
                except Exception as e:
                        error_counter += 1
                        element["status"] = "queued"
                        element["batch_job"] = None
                        prompt_queue.put(element)

                        if error_counter > 30:
                            raise e
                        else:
                            print(f"Retry attempt No. {error_counter}")
                            time.sleep(60)
                            continue
                error_counter = 0
                if is_ready:
                    outputs, positions, erroneous_requests, errors = model.download_batch(batch_name, element["batch_job"])

                    for output, m in zip(outputs, positions):
                        # Process the result and update the element
                        output = flatten_string(output)       
                        idx = element["ids"][m]
                        if prompt_configs["iterative_prompts"][element["iteration"]]["parse_output"] is True:
                            print(f"Parsing output for sample {idx} in iteration {element['iteration']} for {prompt_configs['iterative_prompts'][element['iteration']]['output_name']}")
                            result_df.loc[idx, prompt_configs["iterative_prompts"][element["iteration"]]["output_name"]] = output

                        query_log_df.loc[idx, [f"query_{element['iteration']}", f"response_{element['iteration']}"]] = [json.dumps(element["prompts"][m]), output]

                        if element["iteration"] + 1 < len(prompt_configs["iterative_prompts"]):
                            # Append the output to the last prompt in the list for the next iteration
                            element["prompts"][m][-1]["content"] += output

                    if len(erroneous_requests) > 0:
                        print(f"Removing erroneous requests {erroneous_requests} from the element.")
                        print(errors)
                        element["ids"] = [element["ids"][m] for m in range(len(element["ids"])) if m not in erroneous_requests]
                        element["prompts"] = [element["prompts"][m] for m in range(len(element["prompts"])) if m not in erroneous_requests]
                        element["samples"] = [element["samples"][m] for m in range(len(element["samples"])) if m not in erroneous_requests]
                        element["examples"] = [element["examples"][m] for m in range(len(element["examples"])) if m not in erroneous_requests] 

                    result_df.to_csv(os.path.join(output_dir, "Model_Output.csv"))
                    query_log_df.to_csv(os.path.join(output_dir, "Query_Log.csv"))
                    if element["iteration"] + 1 < len(prompt_configs["iterative_prompts"]):       
                        element["iteration"] += 1
                        element["status"] = "queued"
                        prompt_queue.put(element)
                        with open(queue_path, "wb") as dill_file:
                            dill.dump(prompt_queue, dill_file)
                    else:
                        print(f"Batch {batch_name} completed and all samples processed.")
                        # Remove the element from the queue
                        with open(queue_path, "wb") as dill_file:
                            dill.dump(prompt_queue, dill_file)
                else:
                    # Reinsert the element into the queue for further processing
                    prompt_queue.put(element)  
                    print("Sleep to avoid rate limiting.")
                    time.sleep(60)
                    print(f"Queue size: {prompt_queue.qsize()}")

        return result_df