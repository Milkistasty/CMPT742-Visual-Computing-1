import os
import json

def create_prompts_json(txt_directory, output_json_file):
    prompts = []

    for filename in os.listdir(txt_directory):
        if filename.endswith(".txt"):
            # print(f"Processing: {filename}")  
            with open(os.path.join(txt_directory, filename), 'r', encoding='utf-8') as file:
                prompt_text = file.read().strip()

                parts = filename.split('_')
                if filename.startswith("utkinect"):
                    action = parts[2]
                    frame = parts[-1].replace('.txt', '')
                else:
                    action = parts[0]
                    frame = parts[-1].replace('.txt', '')

                prompt_text = f"{action}, {frame}, {prompt_text}"

                prompt_data = {
                    "source": f"source/{filename.replace('.txt', '.png')}",
                    "target": f"target/{filename.replace('.txt', '.png')}",
                    "prompt": prompt_text
                }

                prompts.append(prompt_data)
                # print(prompt_data)  

    with open(output_json_file, 'w', encoding='utf-8') as json_file:
        for prompt in prompts:
            json_file.write(json.dumps(prompt) + '\n')


txt_directory = 'C:/Users/Alienware/Desktop/github/projectdata/output/'
output_json_file = 'C:/Users/Alienware/Desktop/github/projectdata/prompt.json'

create_prompts_json(txt_directory, output_json_file)
