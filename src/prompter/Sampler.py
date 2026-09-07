from vllm.sampling_params import SamplingParams
import json

def get_samplers(prompt_configs, seed):
    samplers = []
    if "iterative_prompts" in prompt_configs:
        for prompt_config in prompt_configs["iterative_prompts"]:
            samplers.append(SamplingParams(seed=seed, **prompt_config["prompt_args"]))
    else:
        samplers.append(SamplingParams(seed=seed, **prompt_configs["prompt_args"]))

    return samplers


def create_json_template(fields):
    """ Creates a json object structure for guided content creation """
    json_template = {
        "properties": {},
        "required": [],
        "title": "ParserStructure",
        "type": "object"
    }
    for field in fields:
        json_template["properties"][field] = {"title": field, "type": "string"}
        json_template["required"].append(field)
    return json.dumps(json_template)