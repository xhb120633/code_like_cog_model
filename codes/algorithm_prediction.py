# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:24:19 2024

@author: 51027
"""

from openai import OpenAI
import pickle
import json
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt


# Create an OpenAI client with your deepinfra token and endpoint
def classify_algorithm(model, text_description):

    fixed_instruction = """
In a sorting task, human participants sorted 6 unordered images by pair-wise comparisons, without knowing the underlying order. Each comparison will be given a feedback whether the pair is in order or not. The out-of-order pair will swap if they are out of order. 

Human participants are instructed to leave a description about how they complete the task. Given a narrative describing a sorting strategy, classify the sorting algorithm revealed in the narrative. The sorting algorithms to be classified are:

- Gnome Sort
- Selection Sort
- Insertion Sort
- Bubble Sort
- Comb Sort
- Modified Selection Sort
- Shaker Sort
- Successive Sweeps
- Backward Gnome Sort
- Backward Selection Sort
- Backward Insertion Sort
- Backward Bubble Sort
- Backward Comb Sort
- Backward Modified Selection Sort
- Backward Shaker Sort
- Backward Successive Sweeps

If the sorting algorithm revealed in the narrative is not one of the listed algorithms, it should be classified as "Unidentified". If you don't know which algorithm it belongs to, also classify it as "Unidentified".

Please generate the classification based on the narrative provided. Ensure that the output is only the name of the sorting algorithm or "Unidentified". No explainations, examples or clarifications.
"""


    if 'gpt' in model:
        api_key = 'Your OpenAI API key'
        url = 'https://api.openai.com/v1/chat/completions'
    elif 'llama' in model:
        api_key="Your Deepinfra API key"
        url ="https://api.deepinfra.com/v1/openai"
        
    openai = OpenAI(
        api_key=api_key,
        base_url=url,
    ) 
    
    tmp_message =  [{"role": "assistant", "content":fixed_instruction},
        {"role": "user", "content":'The strategy description is ' + text_description}]
    
    response = openai.chat.completions.create(
        model = model,
        messages = tmp_message,
        temperature=0.1,
        max_tokens=6,
        seed = 2024
        )

    return response.choices[0].message.content



# Function to load data from the JSON file
def load_data_from_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Function to save updated data back to the JSON file
def save_data_to_json(filepath, data):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

# Function to load data from the JSON file
def load_data_from_pickle(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)

# Function to save updated data back to the JSON file
def save_data_to_pickle(filepath, data):
    with open(filepath, 'wb') as file:
       pickle.dump(data, file)

strategy_dic = ['Unidentified','Gnome Sort','Selection Sort','Insertion Sort','Bubble Sort','Comb Sort','Modified Selection Sort','Shaker Sort','Successive Sweeps','Backward Gnome Sort','Backward Selection Sort','Backward Insertion Sort','Backward Bubble Sort','Backward Comb Sort','Backward Modified Selection Sort','Backward Shaker Sort','Backward Successive Sweeps']

dir_name= 'C:/Users/51027/Documents/GitHub/sorting_algorithm_text_analysis/data'

participant_data = pd.read_csv(dir_name+ '/participants.csv')
participant_data = np.array(participant_data)

#filter out the duplicate trails
participant_data = participant_data[participant_data[:,5]==False,:]
participant_data = participant_data[participant_data[:,9]==False,:]

column_names = [ 'participant_id','network_id','replication','generation', 'condition','cloned','mean_trial_score','algorithm','algorithm_description','exclusion_flag'
]
participant_data = pd.DataFrame(participant_data, columns=column_names)

## choosing RNN assigned algorithms
participant_data_df = pd.DataFrame(participant_data).set_index('participant_id')
participant_data_df.index = participant_data_df.index.astype(str)

file_path = 'result/generated_codes_data.json'
codes_data = load_data_from_json(file_path)

model_list = ["meta-llama/Meta-Llama-3-70B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct",'gpt-4-0125-preview']
for participant, participant_info in codes_data.items():
    # codes_data[participant]['algorithm_classification']={}
    participant_info['algorithm'] = strategy_dic[int(participant_data_df.at[participant, 'algorithm'])]
    for model in model_list:
        codes_data[participant]['algorithm_classification'][model] = None
        
def classify_algorithms_for_participant(participant,participant_info):
    print(f'Classifying participant {participant}')
    for model in model_list:
        if participant_info['algorithm_classification'][model] is None:
            tmp_algorithm = None
            n_limit = 10
            n = 0
            while tmp_algorithm not in strategy_dic:
                tmp_algorithm = classify_algorithm(model, participant_info['algorithm_description'])
                    
                if tmp_algorithm in strategy_dic:
                    break
                elif n >= n_limit:
                    print(f'Exceeding max allowed attempts for participant {participant}')
                    break
                n += 1
            participant_info['algorithm_classification'][model] = tmp_algorithm
            codes_data[participant] = participant_info
            
            # Assuming save_data_to_json is adapted to handle concurrent access safely
            save_data_to_json(file_path, codes_data)


# Specify the number of threads
max_workers = 100  # Adjust this to control the number of parallel tasks

# Sending parallel requests
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit tasks to classify algorithms for each participant
    futures = [executor.submit(classify_algorithms_for_participant, participant, participant_info) for participant, participant_info in codes_data.items()]

    
#post processing
for participant, participant_info in codes_data.items():
    for model in model_list:
        if codes_data[participant]['algorithm_classification'][model] not in strategy_dic:
            codes_data[participant]['algorithm_classification'][model] = 'Unidentified'
    
        
    
file_path = 'result/generated_codes_data.json'
save_data_to_json(file_path,codes_data)

codes_data = load_data_from_json(file_path)

model_algorithm_accuracy = {model: {algorithm: 0 for algorithm in strategy_dic} for model in model_list}
total_count = {algorithm: 0 for algorithm in strategy_dic}
# Iterate over each participant's data
for participant, data in codes_data.items():
    ground_truth_algorithm = data['algorithm']
    algorithm_classification = data['algorithm_classification']
    total_count[ground_truth_algorithm] += 1
    # Update model accuracy for each algorithm
    for model, prediction in algorithm_classification.items():
        if prediction == ground_truth_algorithm:
            model_algorithm_accuracy[model][ground_truth_algorithm] += 1

# Calculate accuracy for each model on each algorithm
total_participants = len(codes_data)
for model, algorithm_accuracy in model_algorithm_accuracy.items():
    for algorithm, correct_count in algorithm_accuracy.items():
        if total_count[algorithm] != 0:
            model_algorithm_accuracy[model][algorithm] /=  total_count[algorithm]
        else:
            model_algorithm_accuracy[model][algorithm] = 0
        
# Extract algorithm names and model names
algorithms = list(model_algorithm_accuracy[next(iter(model_algorithm_accuracy))].keys())
models = list(model_algorithm_accuracy.keys())

# Set bar width and spacing
bar_width = 0.2
spacing = 0.1
num_models = len(models)

# Set x positions for the bars
x = np.arange(len(algorithms))

# Plot grouped bars for each model
for i, model in enumerate(models):
    # Calculate x position for the model's bars within the group
    model_x = x + (bar_width + spacing) * (i - num_models / 2)

    # Extract accuracies for the current model
    accuracies = [model_algorithm_accuracy[model][algorithm] for algorithm in algorithms]

    # Plot the bars for the current model
    plt.bar(model_x, accuracies, width=bar_width, label=model)

# Set x-axis labels with 45 degree rotation
plt.xticks(x, algorithms, rotation=45, ha='right')

# Add a horizontal line at y = 1/17 (chance level)
plt.axhline(y=1/17, color='red', linestyle='--')

# Add labels and title
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.title('Accuracy of Model Predictions for Different Algorithms')

# Add legend
plt.legend()
# Show plot
plt.tight_layout()
plt.savefig('strategy_classification_comparison.png',bbox_inches='tight', dpi = 300)
plt.show()