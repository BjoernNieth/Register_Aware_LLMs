import os
import csv

class Experiment_Logger_CSV():
    def __init__(self, file_path, prompts_per_sample):
        self.file_path = file_path

        # Create header to store prompts_per_sample responses and prompts
        if prompts_per_sample > 1:
            header = ["id"]
            for i in range(prompts_per_sample):
                header.append(f"prompt_{i+1}")
                header.append(f"response_{i+1}")
        else:
            header = ["id", "prompt", "response"]

        if not os.path.isfile(file_path):
            f = open(file_path, "w")
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()

    def log_prompt(self, content):
        with open(self.file_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(content)
