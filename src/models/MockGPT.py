from openai import OpenAI
import os
import time
import re
import random
import json 
from .Base_Model import Base_Model

#def dummy_gpt(prompt):
#    print(prompt)
#    return "This is a dummy gpt response." 

class MockGPT(Base_Model):
    def __init__(self, model_name, model_args, output_dir):
        self.client = OpenAI()
        self.model_name = model_name
        self.batch_size = 1
        self.call_dir = os.path.join(output_dir, "API_Calls", "Calls")
        self.response_dir = os.path.join(output_dir, "API_Calls", "Responses")
        if not os.path.exists(self.call_dir):
            os.makedirs(self.call_dir)
        if not os.path.exists(self.response_dir):
            os.makedirs(self.response_dir)
        
        self.json_dump = os.path.join(output_dir, "json_dump.json")
        if not os.path.exists(self.json_dump):
            with open(self.json_dump, "w", encoding="utf-8") as f:
                f.write("")
        
        self.mock_response = "This is a mock response from MockGPT."
        self.mock_requests = {}
    
    def chat(self, prompts, sampling_params ):
        raise NotImplementedError("MockGPT does not support chat method.")
    
    def generate(self, prompts, sampling_params):
        raise NotImplementedError("MockGPT does not support generate method.")
    
    def write_batch(self, prompts, sampling_params, batch_name):
        """For now assume single batch with single prompt."""
        request = []
        for m in range(len(prompts)):
            task = {
                "custom_id": f"{batch_name}_{m}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": f"{self.model_name}",
                    "max_tokens": sampling_params.max_tokens,
                    "messages": prompts[m]
                }
            }
            request.append(task)
        with open(os.path.join(self.call_dir, f"{batch_name}.jsonl"), "w", encoding="utf-8") as f:
            for task in request:
                f.write(json.dumps(task) + "\n")

    def query_batch(self, batch_name):
        """For now assume single batch with single prompt."""
        print(f"Querying batch {batch_name}...")
        batch_file_path = os.path.join(self.call_dir, f"{batch_name}.jsonl")
        print(open(batch_file_path, "rb"))
        lines = 0
        with open(batch_file_path, "rb") as f:
            for _ in f:
                lines += 1
        print(f"MockGPT: {lines} requests in batch {batch_name}.")
        self.mock_requests[batch_name + "_batch_job_id"] = lines
        return batch_name + "_batch_job_id"  # Mocking the batch job ID for testing purposes
    
    def batch_completed(self, batch_job):
        """Check if the batch job is completed."""
        if random.choice([True, False, False, False, False, False]):
            print(f"Batch job {batch_job} is completed.")
            return True
        else:
            print(f"Batch job {batch_job} is not completed yet.")
            return False

    def download_batch(self, batch_name, batch_job):
        print(self.mock_requests.keys())
        print(f"Downloading batch {batch_name} with job ID {batch_job}...")
        #erronous_requests = [0]
        #error = ["Test error message for batch processing."]
        return [self.mock_response] * self.mock_requests[batch_job] , list(range(self.mock_requests[batch_job])), [], []
                