from .Base_Model import Base_Model
from vllm import LLM
import time
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download


class Qwen_Model(Base_Model):
    def __init__(self, model_name, model_args):
        self.batch_size = model_args["max_num_seqs"]
        
        # Manually download the model from Hugging Face Hub because vLLM has issues with models which are stored in Meta backend
        snapshot_download(
            repo_id=model_name,
            local_dir_use_symlinks=False,
            resume_download=True,
            # Limit the number of workers to avoid exceeding the maximum number of open files
            max_workers=16
        )
        print("Qwen like model downloaded from Hugging Face Hub.")
        #self.model = LLM(model=model_name, dtype=torch.bfloat16,  **model_args)
        # Check if enable_thinking is set in the config
        if "enable_thinking" in model_args:
            enable_thinking = model_args["enable_thinking"]
            del model_args["enable_thinking"]
        # If not, disable thinking mode
        else:
            enable_thinking = False
        print(f"Running Qwen model with enable_thinking: {enable_thinking}")
        self.model = LLM(model=model_name,  tokenizer=model_name,
                        chat_template_kwargs={"enable_thinking": enable_thinking},
                        **model_args)

    def chat(self, prompts, sampling_params):
        # Use model.generate() to get generated token IDs
        start_time = time.time()
        output = self.model.chat(
            prompts,
            sampling_params
        )

        tokens_generated = sum(len(o.outputs[0].token_ids) for o in output)
        end_time = time.time()
        print(f"Speed: " + str(tokens_generated/(end_time - start_time)) + " Token/sec")
        return output
    
    def generate(self, prompts, sampling_params):
        # Use model.generate() to get generated token IDs
        start_time = time.time()
        output = self.model.generate(
            prompts,
            sampling_params
        )

        tokens_generated = sum(len(o.outputs[0].token_ids) for o in output)
        end_time = time.time()
        print(f"Speed: " + str(tokens_generated/(end_time - start_time)) + " Token/sec")
        return output
