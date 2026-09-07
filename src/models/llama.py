from .Base_Model import Base_Model
from vllm import LLM
import time
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download


class Llama_Model(Base_Model):
    def __init__(self, model_name, model_args, output_dir):
        #self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/" + model_name)
        mp.set_start_method("spawn", force=True)
        self.batch_size = model_args["max_num_seqs"]
        
        # Manually download the model from Hugging Face Hub because vLLM has issues with models which are stored in Meta backend
        snapshot_download(
            repo_id=model_name,
            local_dir_use_symlinks=False,
            resume_download=True,
            # Limit the number of workers to avoid exceeding the maximum number of open files
            max_workers=16
        )
        print("Llama like model downloaded from Hugging Face Hub.")
        #self.model = LLM(model=model_name, dtype=torch.bfloat16,  **model_args)
        self.model = LLM(model=model_name,  model_impl="vllm",  **model_args)

    def chat(self, prompts, sampling_params):
        # Use model.generate() to get generated token IDs
        print(sampling_params)
        start_time = time.time()
        outputs = self.model.chat(
            prompts,
            sampling_params
        )

        tokens_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        end_time = time.time()
        print(f"Speed: " + str(tokens_generated/(end_time - start_time)) + " Token/sec")
        for o in outputs:
            assert len(o.outputs[0].token_ids) >= sampling_params.min_tokens, f"The model generated fewer than {sampling_params.min_tokens} tokens. Please raises the context length."
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
