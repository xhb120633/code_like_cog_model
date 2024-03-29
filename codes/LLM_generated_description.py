# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 14:40:19 2024
Test LLM inductive bias in sorting algorithms

@author: 51027
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
import json
from concurrent.futures import ThreadPoolExecutor,as_completed
import pickle
import itertools
import random

def get_description(algorithm):

    fixed_instruction = """
Imagine you are a human participant in a sorting task. In this task, there are trials of unknown lists of images in length of six. There is no relationship between the images and orders, and the images and orders are completely random on each trial. Participants need to conduct pairwise comparisons to indicate whether two selected images are in order or out of order. If they are out of order, they will be swamped. Participants can do unlimited times of comparisons but will need to click 'finish' when they think they have done sorting.
Now, you are instructed to describe how you did this task for future participants to refer. Suppose you are using one sorting algorithm, and please describe how you did the task with the algorithm. 
You should not mention the algorithm name, nor try to introduce the background of any sorting algorithm. You only need to integrate the task context and describe to future participants how you did the task by using the sorting algorithm. You should talk naturally and directly start with your description. No need to respond verbally to the instructions.If the algorithm contains backward, it means you should start from right of the list; otherwise, you should always start from left of the list. You shold make the content a bit brief since humans do not like speaking too much on the same thing.
"""


    
    tmp_message =  [{"role": "assistant", "content":fixed_instruction},
        {"role": "user", "content":'The algorith, you have been using is ' + algorithm}]
    
    tmp_response = openai.ChatCompletion.create(
        model = 'gpt-4-turbo-preview',
        messages = tmp_message,
        temperature = 0.7,
        max_tokens = 300
        )
    
    response = tmp_response['choices'][0]['message']['content']
    return response


def get_codes(text_description):

    fixed_instruction = """
Create a Python sorting function 'human_like_sorting',which is directly executable and effective for a general sorting task, based on a strategic approach to sorting a list of items. The function should simulate the decision-making process humans might use when sorting, without direct comparison to a known 'true order'. The function must:

1. Accept 'initial_stimuli' as a list of items to be sorted, with 'true_order' available only to the internal 'attempt_swap' function for providing feedback on swap actions. 'True_order' represents the correct order of items and is used to simulate feedback on whether a swap action brings items closer to their desired order:

    def attempt_swap(current_order, true_order, action):
        index1, index2 = action
        if 0 <= index1 < len(current_order) and 0 <= index2 < len(current_order) and index1 != index2:
            item1, item2 = current_order[index1], current_order[index2]
            true_pos1, true_pos2 = true_order.index(item1), true_order.index(item2)
            if (true_pos1 > true_pos2 and index1 < index2) or (true_pos1 < true_pos2 and index1 > index2):
                current_order[index1], current_order[index2] = current_order[index2], current_order[index1]
                return True, current_order
        return False, current_order

2. Return a log of actions attempted, regardless of the outcome, formatted as tuples of item indices attempted for swapping, as well as the outcome of this attempt. A 'finish' action signals the end of the sorting process. This log tracks the sorting actions as well as the outcome of the comparison, simulating a step-by-step decision-making process.

3. Contain all necessary logic for sorting within the function, making it capable of sorting lists of any length using basic operations. The function should not depend on external libraries or Python's built-in sorting functions for its core sorting logic. To avoid inifinte loop, the function should set a hard limit up to 50 comparison attempts and always ended up with a 'finish'. All the basic functions should be correctly used (if used).

4. Refrain from using 'true_order' for direct comparisons with 'current_order' to assess sorting completion. Instead, infer the completion through the sorting process itself, akin to how a person might determine they have finished sorting without knowing the exact 'true_order'. 'true_order' can only be used in the 'attempt_swamp' function.

Focus on generating executable, valid Python code that effectively sorts 'initial_stimuli' into order. The code should first and foremost be functional and capable of performing the sorting task well under general conditions. Once functional validity is ensured, align the code as closely as possible with the strategic description provided, within the bounds of simulating human-like sorting behavior under fair computational constraints.

Please provide only the implemented function, ready for direct execution with 'initial_stimuli' and 'true_order' inputs. Don't include any comments, explainations, notes or examples.
"""


    
    tmp_message =  [{"role": "assistant", "content":fixed_instruction},
        {"role": "user", "content":'The strategy description is ' + text_description}]
    
    response = openai.ChatCompletion.create(
        model = 'gpt-4-turbo-preview',
        messages = tmp_message,
        temperature=0.1,
        max_tokens=1024,
        seed = 2024
        )

    return response

def generate_and_save_code(participant_id):
    info = recovered_codes_data[participant_id]
    if info['generated_code'] is None:  # Check if code needs to be generated
        print(f"Generating code for participant {participant_id}...")
        description = info['algorithm_description']
        tmp_response = get_codes(description)  # Assuming get_codes is an I/O-bound function
        generated_code = tmp_response['choices'][0]['message']['content']


        recovered_codes_data[participant_id]['generated_code'] = generated_code
        
        # Assuming save_data_to_json is adapted to handle concurrent access safely
        save_data_to_json(file_path, recovered_codes_data)

# Function to request and save descriptions for a specific algorithm, repeated 100 times
def request_descriptions_for_algorithm(algorithm_name, sample_id_start, sample_size=100):
    with ThreadPoolExecutor(max_workers=100) as executor:
        # Submit description requests for the current algorithm, ensuring unique sub_ids
        futures = {executor.submit(get_description, algorithm_name): sample_id for sample_id in range(sample_id_start, sample_id_start + sample_size)}
        
        descriptions = {}
        for future in as_completed(futures):
            sub_id = futures[future]
            try:
                description = future.result()
            except Exception as exc:
                print(f'Description request generated an exception: {exc}')
            else:
                # Using 'algorithm_description' and 'true_algorithm' as keys
                descriptions[sub_id] = {'algorithm_description': description, 'true_algorithm': algorithm_name}
                
    return descriptions

# Main function to generate descriptions for all algorithms with unique sub_ids
def generate_descriptions_for_all_algorithms():
    all_descriptions = {}
    sample_id_counter = itertools.count()  # Infinite counter for unique sample IDs
    
    for algorithm_name in standard_sorting_algorithms.keys():
        print(f"Requesting descriptions for {algorithm_name}...")
        sample_id_start = next(sample_id_counter) * 100  # Ensure a unique starting point for each algorithm
        descriptions_for_algorithm = request_descriptions_for_algorithm(algorithm_name, sample_id_start)
        all_descriptions.update(descriptions_for_algorithm)
    
    return all_descriptions
        
def get_embeddings(text, model): 
    response = openai.Embedding.create(
            input=text,
            engine= model) 
    return response

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


openai.api_key = 'XXX' # Your Own API

dir_name= 'C:/Users/51027/Documents/GitHub/sorting_algorithm_text_analysis/data'

strategy_dic = ['Unidentified','Gnome Sort','Selection Sort','Insertion Sort','Bubble Sort','Comb Sort','Modified Selection Sort','Shaker Sort','Successive Sweeps','Backward Gnome Sort','Backward Selection Sort','Backward Insertion Sort','Backward Bubble Sort','Backward Comb Sort','Backward Modified Selection Sort','Backward Shaker Sort','Backward Successive Sweeps']


participant_data = pd.read_csv(dir_name+ '/participants.csv')
participant_data = np.array(participant_data)

#filter out the duplicate trails
participant_data = participant_data[participant_data[:,5]==False,:]
participant_data = participant_data[participant_data[:,9]==False,:]

column_names = [ 'participant_id','network_id','replication','generation', 'condition','cloned','mean_trial_score','algorithm','algorithm_description','exclusion_flag'
]
participant_data = pd.DataFrame(participant_data, columns=column_names)


network_data = pd.read_csv(dir_name+ '/networks.csv')
network_data = np.array(network_data)

stimulus_data = pd.read_csv(dir_name+ '/orderings.csv')
stimulus_data= np.array(stimulus_data)

summary_behavioral_data = pd.read_csv(dir_name+ '/trials.csv')
summary_behavioral_data = np.array(summary_behavioral_data )

raw_behavioral_data = pd.read_csv(dir_name+ '/comparisons.csv')
column_names =['participant_id','network_id','generation','condition','mean_trial_score','cloned','replication','trial_index','comparison_index','image_i_position_index','image_j_position_index','swap','rank_image_i','rank_image_j']
raw_behavioral_data = pd.DataFrame(raw_behavioral_data, columns = column_names)
raw_behavioral_data = raw_behavioral_data.loc[raw_behavioral_data['cloned'] == False]

    
file_path = 'result/recovered_codes_data.json'
original_file_path = 'result/generated_codes_data.json'
codes_data = load_data_from_json(original_file_path)
# Compute the overall list of lengths for all participants' descriptions
lengths = [len(participant['algorithm_description']) for participant in codes_data.values()]

# Now, plot a histogram of these lengths
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Description Lengths')
plt.xlabel('Length of Algorithm Descriptions')
plt.ylabel('Frequency')
plt.show()


save_file_path = 'F:/sorting_algorithm_data/standard_algorithms_simulation.pkl'

standard_sorting_algorithms = load_data_from_pickle(save_file_path)
## conduct a small experiment to see how this performs

# #randomly select 100 participants to run the code generation and execution
# sampled_participants = participant_data['participant_id'].sample(n=100, random_state=2024)

# Specify the number of threads
max_workers = 100 # Adjust this to control the number of parallel tasks

# Generate descriptions
recovered_codes_data = generate_descriptions_for_all_algorithms()
# for participant_id in recovered_codes_data:
#     recovered_codes_data[participant_id]['generated_code'] = None
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(generate_and_save_code, participant_id) for participant_id in recovered_codes_data]