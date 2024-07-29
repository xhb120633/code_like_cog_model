# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:03:46 2024

@author: 51027
"""


import pandas as pd
import numpy as np
from openai import OpenAI
import json
from concurrent.futures import ThreadPoolExecutor

def get_codes(model, text_description):

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

2. The attempt_swap function should be defined within human_like_sorting, so calling attempt_swap from within human_like_sorting is sufficient to run simulations.

3. Return a log of actions attempted, regardless of the outcome, formatted as tuples of item indices attempted for swapping, as well as the outcome of this attempt. A 'finish' action signals the end of the sorting process. This log tracks the sorting actions as well as the outcome of the comparison, simulating a step-by-step decision-making process.

4. Contain all necessary logic for sorting within the function, making it capable of sorting lists of any length using basic operations. The function should not depend on external libraries or Python's built-in sorting functions for its core sorting logic. To avoid inifinte loop, the function should set a hard limit up to 50 comparison attempts and always ended up with a 'finish'. All the basic functions should be correctly used (if used).

5. Refrain from using 'true_order' for direct comparisons with 'current_order' to assess sorting completion. Instead, infer the completion through the sorting process itself, akin to how a person might determine they have finished sorting without knowing the exact 'true_order'. 'true_order' can only be used in the 'attempt_swamp' function.

Focus on generating executable, valid Python code that effectively sorts 'initial_stimuli' into order. The code should first and foremost be functional and capable of performing the sorting task well under general conditions. Once functional validity is ensured, align the code as closely as possible with the strategic description provided, within the bounds of simulating human-like sorting behavior under fair computational constraints.

Please provide only the implemented function, ready for direct execution with 'initial_stimuli' and 'true_order' inputs. Don't include any comments, explainations, notes or examples.
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
        max_tokens=1024,
        seed = 2024
        )

    return response

def get_codes_icl(model, text_description,example_sequence):

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

2. The attempt_swap function should be defined within human_like_sorting, so calling attempt_swap from within human_like_sorting is sufficient to run simulations.

3. Return a log of actions attempted, regardless of the outcome, formatted as tuples of item indices attempted for swapping, as well as the outcome of this attempt. A 'finish' action signals the end of the sorting process. This log tracks the sorting actions as well as the outcome of the comparison, simulating a step-by-step decision-making process.

4. Contain all necessary logic for sorting within the function, making it capable of sorting lists of any length using basic operations. The function should not depend on external libraries or Python's built-in sorting functions for its core sorting logic. To avoid inifinte loop, the function should set a hard limit up to 50 comparison attempts and always ended up with a 'finish'. All the basic functions should be correctly used (if used).

5. Refrain from using 'true_order' for direct comparisons with 'current_order' to assess sorting completion. Instead, infer the completion through the sorting process itself, akin to how a person might determine they have finished sorting without knowing the exact 'true_order'. 'true_order' can only be used in the 'attempt_swamp' function.

Focus on generating executable, valid Python code that effectively sorts 'initial_stimuli' into order. The code should first and foremost be functional and capable of performing the sorting task well under general conditions. Once functional validity is ensured, align the code as closely as possible with the strategic description provided, within the bounds of simulating human-like sorting behavior under fair computational constraints.

Please provide only the implemented function, ready for direct execution with 'initial_stimuli' and 'true_order' inputs. Don't include any comments, explainations, notes or examples.
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
        {"role": "user", "content":'The strategy description is ' + text_description + 'The action sequence in an example trial is' + example_sequence}]
    
    response = openai.chat.completions.create(
        model = model,
        messages = tmp_message,
        temperature=0.1,
        max_tokens=1024,
        seed = 2024
        )

    return response

def generate_and_save_code(model, participant_id):
    info = codes_data[participant_id]
    if info['generated_code'] is None:  # Check if code needs to be generated
        print(f"Generating code for participant {participant_id}...")
        description = info['algorithm_description']
        tmp_response = get_codes(model,description)  # Assuming get_codes is an I/O-bound function
        generated_code = tmp_response.choices[0].message.content


        codes_data[participant_id]['generated_code'] = generated_code
        
        # Assuming save_data_to_json is adapted to handle concurrent access safely
        save_data_to_json(file_path, codes_data)
        
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


# openai.api_key =  # YOUR OWN API

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

tmp_df = participant_data[['participant_id','algorithm_description']]

# Convert the DataFrame to a dictionary format suitable for JSON
data_to_save = tmp_df.set_index('participant_id').to_dict(orient='index')

# Initialize each participant's entry with a placeholder for the generated code
for participant_id in data_to_save:
    data_to_save[participant_id]['generated_code'] = None

model = "codellama/CodeLlama-34b-Instruct-hf"
model_save_name = 'CodeLlama-34b-Instruct-hf'
file_path = f'result/generated_codes_data_{model_save_name}.json'

# Save this initial data to a JSON file
save_data_to_json(file_path, data_to_save)
    
codes_data = load_data_from_json(file_path)

# conduct a small experiment to see how this performs

#randomly select 100 participants to run the code generation and execution 
sampled_participants = participant_data['participant_id'].sample(n=100, random_state=2024)

# Specify the number of threads
max_workers = 100 # Adjust this to control the number of parallel tasks

# Sending parallel requests on API
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(generate_and_save_code, model, participant_id) for participant_id in codes_data]