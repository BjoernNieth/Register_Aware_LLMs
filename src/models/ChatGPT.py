from openai import OpenAI
import os
import time
import json 
from .Base_Model import Base_Model

#def dummy_gpt(prompt):
#    print(prompt)
#    return "This is a dummy gpt response." 

class ChatGPT_API(Base_Model):
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
            
    
    def chat(self, prompts, sampling_params ):
        assert len(prompts) == 1, "We use the OpenAI API only for single prompts."

        
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=prompts[0],
            max_completion_tokens=sampling_params.max_tokens,
        )
        print("Max output tokens:", sampling_params.max_tokens)
        #response = dummy_gpt(prompts[0])
        print(response)
        with open(self.json_dump, "a", encoding="utf-8") as f:
            json.dump(response.to_dict(), f, ensure_ascii=False, indent=4)
        #return [response.output_text]
        return [response.choices[0].message.content]
    
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
        print(f"Querying batch {batch_name}...")
        batch_file_path = os.path.join(self.call_dir, f"{batch_name}.jsonl")
        batch_file = self.client.files.create(
            file=open(batch_file_path, "rb"),
            purpose="batch"
        )
        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        return batch_job
    
    def batch_completed(self, batch_job):
        """Check if the batch job is completed."""
        batch_job = self.client.batches.retrieve(batch_job.id)
        if batch_job.status == "failed" or batch_job.status == "cancelled" or batch_job.status == "expired":
            raise Exception(f"Batch job {batch_job.id} failed: {batch_job.errors}")
        elif batch_job.status == "completed":
            return True
        else: 
            return False

    def download_batch(self, batch_name, batch_job):
        try:
            output_file_id = self.client.batches.retrieve(batch_job.id).output_file_id
            if output_file_id is None:
                raise Exception(f"Batch job {batch_job.id} has no output file.")
            
            result = self.client.files.content(output_file_id).content
            with open(os.path.join(self.response_dir, batch_name + ".jsonl"), 'wb') as file:
                file.write(result)
            
            results = []
            with open(os.path.join(self.response_dir, batch_name + ".jsonl"), 'r') as file:
                for line in file:
                    # Parsing the JSON string into a dict and appending to the list of results
                    json_object = json.loads(line.strip())
                    results.append(json_object)
            positions = []
            outputs = []  
            erroneous_requests = []
            errors = []
            for result in results:
                if result["error"] is not None:
                    print(f"Error in request {result['custom_id']}: {result['error']}")
                    erroneous_requests.append(result["custom_id"])
                    errors.append(result["error"])
                else:
                    positions.append(int(result['custom_id'].replace(batch_name + "_", "")))
                    outputs.append(result['response']['body']['choices'][0]['message']['content'])  
                
            return outputs, positions, erroneous_requests, errors
        
        except Exception as e:
            print(f"Error downloading batch {batch_name}: {e}")
            batch_job_latest = self.client.batches.retrieve(batch_job.id)
            print(f"Batch job status: {batch_job_latest.status}")
            if batch_job_latest.error_file_id is not None:
                print(f"Following error occured in the batch job {batch_job.id}:")
                print(self.client.files.content(batch_job_latest.error_file_id).content)
            print(batch_job)
            raise e