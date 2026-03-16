from .Base_Model import Base_Model
from vllm import LLM
import time
import torch.multiprocessing as mp
import torch



class Mixtral_Model(Base_Model):
    def __init__(self, model_name, model_args, output_dir):
        self.batch_size = model_args["max_num_seqs"]
        print("Mixtral like model is being downloaded from Hugging Face Hub. This may take a while...")
        #self.model = LLM(model=model_name, dtype=torch.bfloat16,  **model_args)
        self.model = LLM(model=model_name,
            trust_remote_code=True,
            **model_args
        )

    def __remove_prefix_flags(self, prompts):
        """
        Removes the "prefix" flag from assistant messages in the prompts.
        This is necessary to avoid issues with a prefix flag which occurs not as the last message in the prompt for Mixtral models.
        """
        cleaned_prompts = []
        for prompt in prompts:
            cleaned_prompt = []
            for message in prompt:
                # Copy message to avoid modifying input in-place
                msg_copy = message.copy()
                if msg_copy.get("role") == "assistant" and "prefix" in msg_copy:
                    msg_copy.pop("prefix")
                cleaned_prompt.append(msg_copy)
            cleaned_prompts.append(cleaned_prompt)
        return cleaned_prompts
    
    def chat(self, prompts, sampling_params):
        cleand_prompts = self.__remove_prefix_flags(prompts)
        start_time = time.time()
        outputs = self.model.chat(
            cleand_prompts,
            sampling_params
        )

        tokens_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        end_time = time.time()
        print(f"Speed: " + str(tokens_generated/(end_time - start_time)) + " Token/sec")
        return [output.outputs[0].text for output in outputs]
    
    def generate(self, prompts, sampling_params):
        # Use model.generate() to get generated token IDs
        start_time = time.time()
        outputs = self.model.generate(
            prompts,
            sampling_params
        )

        tokens_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        end_time = time.time()
        print(f"Speed: " + str(tokens_generated/(end_time - start_time)) + " Token/sec")
        return [output.outputs[0].text for output in outputs]
