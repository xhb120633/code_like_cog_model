# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:41:15 2024

@author: 51027
"""

import json
import pandas as pd
import numpy as np
import re
import itertools
from openai import OpenAI
import random
import signal
from contextlib import contextmanager
import pickle

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        
        
def prepare_codes_data(codes_data, stimulus_data):
    """
    Adjusts codes_data to include 'trials' under 'real_task_simulation' for each participant,
    based on stimulus data, setting the stage for simulations. This version only processes participants
    included in codes_data.

    Parameters:
    - codes_data: Dictionary containing participant codes and descriptions.
    - stimulus_data: DataFrame with detailed stimulus data across multiple trials.
    """
    # Iterate over each participant in codes_data
    for participant_id_str in codes_data.keys():
        # Convert participant_id_str back to its original type as in stimulus_data, assuming it's an integer
        participant_id = int(participant_id_str)

        # Filter stimulus_data for the current participant
        participant_stimulus_data = stimulus_data[stimulus_data['participant_id'] == participant_id]

        # Group by 'trial_index' since we are interested in trials for this participant
        for trial_index, trial_data in participant_stimulus_data.groupby('trial_index'):
            # Sort by 'image_index' to maintain the display order
            trial_data_sorted = trial_data.sort_values(by='image_index')

            # Extract 'initial_stimuli' and 'true_order'
            initial_stimuli = trial_data_sorted[trial_data_sorted['state'] == 'final']['image_rank'].tolist()
            true_order = trial_data_sorted[trial_data_sorted['state'] == 'initial']['image_rank'].tolist()

            # Initialize 'real_task_simulation' and 'trials' if not present
            if 'real_task_simulation' not in codes_data[participant_id_str]:
                codes_data[participant_id_str]['real_task_simulation'] = {'trials': {}}

            # Update 'trials' within 'real_task_simulation' with initial stimuli and true order for each trial
            codes_data[participant_id_str]['real_task_simulation']['trials'][str(trial_index)] = {
                'initial_stimuli': initial_stimuli,
                'true_order': true_order
            }

            # Ensure other keys like 'algorithm_description' and 'generated_code' are maintained
            # This step may involve copying existing data or ensuring these keys are not overwritten

    return codes_data

def clean_get_function(code_str):
     # Extract the Python code block from the text
     code_cleaned_match = re.search(r"```python\n(.*?)\n```", code_str, re.DOTALL)
     
     if code_cleaned_match:
         code_cleaned = code_cleaned_match.group(1).strip()
     else:
         print(f"No executable code found for participant {participant_id}")
         return
 
     # Execute the cleaned code
     local_namespace = {}
     try:
         exec(code_cleaned, globals(), local_namespace)
     except Exception as e:
         print(f"Error executing code for participant {participant_id}: {e}")
         return
 
     sort_function = local_namespace.get('human_like_sorting')
     if not sort_function:
         print(f"Sorting function not found in the code for participant {participant_id}")
         return
     
     return sort_function

def pre_exam_for_participant(model, participant_id, codes_data):
    """
    Exam the codes before large-scale simulations. Catch and fix the code bugs within GPT-4 closed-loop.

    Parameters:
    - participant_id: The ID of the participant.
    - participant_data: Dictionary containing all participants' data, including codes and trial info.
    """
    participant_info = codes_data.get(participant_id)
    print(f"Simulating participant {participant_id}")
    if not participant_info:
        print(f"No data found for participant {participant_id}")
        return

    code_str = participant_info.get('generated_code', '')
    description = participant_info.get('algorithm_description', '')
    
    example_stimuli = [3,1,2,4,5,6]
    example_true_order = [1,2,3,4,5,6]
    
    attempts = 0
    max_attempts = 5
    
    ##pre-exam any potential errors in the code and send back to GPT-4 for correction
    corrected_code = code_str
    while attempts < max_attempts:
        sort_function = clean_get_function(corrected_code)
        
        if sort_function is None:
            print(f"Function extraction failed for participant {participant_id}.")
            break

        
        try:
            action_sequence = sort_function(example_stimuli, example_true_order)
            if attempts > 0:
                print("Successful correction!")
            break  # Success, exit the loop
        # # This code is commented since can only work in Linux to break the thread of infinite loop
        # except TimeoutException as t: 
        #     print(f"Attempt {attempts + 1}: Infinite loop during simulation for participant {participant_id}: {t}")
        #     error = 'The code is running out of time and likely stuck in infinite loop!'
        #     corrected_code = automantic_codes_correction(model,description, corrected_code, error)
        #     if corrected_code is None:
        #         print(f"Failed to get correction from (model).")
        #         return
        #     attempts += 1
        except Exception as e:
            print(f"Attempt {attempts + 1}: Error during simulation for participant {participant_id}: {e}")
            # Request correction from GPT-4
            corrected_code = automantic_codes_correction(model, description, corrected_code, str(e))
            if corrected_code is None:
                print("Failed to get correction from GPT-4.")
                break
            attempts += 1
        
            if attempts == max_attempts:
                break
                print("Maximum correction attempts reached.")
                
    if action_sequence is None or len(action_sequence) == 0:
        print(f'Codes not generating behavior, participant:{participant_id}')
                
    ## saving corrected codes and continuing formal analysis
    participant_info['generated_code'] = corrected_code
    codes_data[participant_id] = participant_info
    return codes_data


    
def execute_simulations_for_participant(participant_id, codes_data):
    """
    Executes simulations for all trials of a specific participant, using the extracted sorting function.
    Updates each trial with the simulated behavior sequence and also runs comprehensive simulations
    for all permutations of a 6-length sequence.

    Parameters:
    - participant_id: The ID of the participant.
    - participant_data: Dictionary containing all participants' data, including codes and trial info.
    """
    participant_info = codes_data.get(participant_id)
    print(f"Simulating participant {participant_id}")
    if not participant_info:
        print(f"No data found for participant {participant_id}")
        return

    code_str = participant_info.get('generated_code', '')
    sort_function = clean_get_function(code_str)
    
    # Iterate over trials and execute the sorting function with trial-specific inputs
    for trial_id, trial_info in participant_info.get('real_task_simulation', {}).get('trials', {}).items():
        initial_stimuli = trial_info['initial_stimuli']
        true_order = trial_info['true_order']
        
        # Execute the sorting function and capture the action sequence
        try:
            action_sequence = sort_function(initial_stimuli, true_order)
            trial_info['simulated_behavior'] = action_sequence
        except Exception as e:
            print(f"Error during simulation for participant {participant_id}, trial {trial_id}: {e}")

    # Comprehensive Simulations
    sequence = [1, 2, 3, 4, 5, 6]
    all_permutations = list(itertools.permutations(sequence))
    comprehensive_simulations = {}

    for idx, permutation in enumerate(all_permutations, start=1):
        try:
            # Execute the sorting function for each permutation
            simulated_behavior = sort_function(list(permutation), sequence)
            comprehensive_simulations[str(idx)] = simulated_behavior
        except Exception as e:
            print(f"Error during comprehensive simulation for permutation {idx}: {e}")

    # Store the comprehensive simulations
    if 'comprehensive_simulations' not in participant_info:
        participant_info['comprehensive_simulations'] = comprehensive_simulations
    else:
        participant_info['comprehensive_simulations'].update(comprehensive_simulations)
    
    codes_data[participant_id] = participant_info

def execute_standard_simmulation_for_participant(participant_id, standard_codes_data,standard_algorithms):
    """
    Executes simulations for all trials of a specific participant, using the extracted sorting function.
    Updates each trial with the simulated behavior sequence and also runs comprehensive simulations
    for all permutations of a 6-length sequence.

    Parameters:
    - participant_id: The ID of the participant.
    - participant_data: Dictionary containing all participants' data, including codes and trial info.
    """
    participant_info = standard_codes_data.get(participant_id)
    print(f"Simulating participant {participant_id}")
    if not participant_info:
        print(f"No data found for participant {participant_id}")
        return
    
    for algorithm in standard_algorithms.keys():
        code_str = standard_algorithms[algorithm].get('generated_code', '')
        sort_function = clean_get_function(code_str)
        
        # Iterate over trials and execute the sorting function with trial-specific inputs
        for trial_id, trial_info in participant_info.get('real_task_simulation', {}).get('trials', {}).items():
            initial_stimuli = trial_info['initial_stimuli']
            true_order = trial_info['true_order']
            
            # Execute the sorting function and capture the action sequence
            try:
                action_sequence = sort_function(initial_stimuli, true_order)
                
                # Create the 'standard_simulation' key if it doesn't exist and assign the action_sequence
                trial_info.setdefault('standard_simulation', {}).update({algorithm: {'simulated_behavior': action_sequence}})
            except Exception as e:
                print(f"Error during simulation for participant {participant_id}, trial {trial_id}: {e}")
    
     
    
def automantic_codes_correction(model,text_description, original_codes, errors):

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

    prompt = (
        f"The strategy description is:\n{text_description}\n\n"
        f"The original code is:\n```python\n{original_codes}\n```\n\n"
        f"The error encountered during execution is:\n{errors}\n\n"
        "Corrected code:"
    )
    
    tmp_message =  [{"role": "assistant", "content":fixed_instruction},
        {"role": "user", "content":prompt}]
    
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
    
    response = openai.chat.completions.create(
        model = model,
        messages = tmp_message,
        temperature=0.1,
        max_tokens=1024,
        seed = 2024
        )
    
    corrected_codes = response.choices[0].message.content
    return corrected_codes

def update_codes_data_with_true_behavior(codes_data, raw_behavioral_data):
    """
    Updates codes_data with true comparison behaviors and outcomes from raw_behavioral_data for included participants,
    formatted as a list of tuples. Each tuple includes the comparison indices and the outcome, with a 'finish' action appended.
    
    Parameters:
    - codes_data: Dictionary containing all participants' data, including generated code and trial info.
    - raw_behavioral_data: DataFrame with detailed behavioral data across multiple trials.
    """
    # Convert participant IDs in codes_data to strings for consistent comparison
    codes_data_participants = set(map(str, codes_data.keys()))
    
    # Filter raw_behavioral_data for participants present in codes_data
    filtered_data = raw_behavioral_data[raw_behavioral_data['participant_id'].astype(str).isin(codes_data_participants)]
    
    # Group the filtered behavioral data by participant and trial
    grouped_data = filtered_data.groupby(['participant_id', 'trial_index'])
    
    for (participant_id, trial_index), group in grouped_data:
        # Format the comparison behaviors and outcomes as a list of tuples
        comparisons = [
            ((row['image_i_position_index'], row['image_j_position_index']), row['swap']) 
            for _, row in group.iterrows()
        ]
        comparisons.append(['finish'])  # Append 'finish' to indicate the end of comparisons
        
        participant_id_str = str(participant_id)
        trial_index_str = str(trial_index)
        
        # Ensure the trial exists in codes_data and update with true behavior
        if trial_index_str in codes_data[participant_id_str]['real_task_simulation']['trials']:
            codes_data[participant_id_str]['real_task_simulation']['trials'][trial_index_str]['true_behavior'] = comparisons
        else:
            # Optionally handle missing trials or log a message
            print(f"Missing trial {trial_index_str} for participant {participant_id_str} in codes_data.")


            
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

dir_name= 'C:/Users/51027/Documents/GitHub/sorting_algorithm_text_analysis/data'

strategy_dic = ['Unidentified','gnome sort','selection sort','insert sort','bubble sort','comb sort','modified selection sort','shaker sort','successive sweeps']

# openai.api_key = 'xxx' # YOUR OWN API


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
column_names = ["participant_id","network_id","generation","condition","mean_trial_score","cloned","replication","trial_index","state","image_index","image_rank"]
stimulus_data = pd.DataFrame(stimulus_data, columns = column_names)

summary_behavioral_data = pd.read_csv(dir_name+ '/trials.csv')
summary_behavioral_data = np.array(summary_behavioral_data )

raw_behavioral_data = pd.read_csv(dir_name+ '/comparisons.csv')
column_names =['participant_id','network_id','generation','condition','mean_trial_score','cloned','replication','trial_index','comparison_index','image_i_position_index','image_j_position_index','swap','rank_image_i','rank_image_j']
raw_behavioral_data = pd.DataFrame(raw_behavioral_data, columns = column_names)
raw_behavioral_data = raw_behavioral_data.loc[raw_behavioral_data['cloned'] == False]

model = "codellama/CodeLlama-34b-Instruct-hf"
model = 'gpt-4-0125-preview'
model_save_name = 'CodeLlama-34b-Instruct-hf'
file_path = f'result/generated_codes_data_{model_save_name}.json'

codes_data = load_data_from_json(file_path)

##pre-exam the code first
for participant_id, info in codes_data.items():
     pre_exam_for_participant(model, participant_id, codes_data)

     
save_data_to_json(file_path, codes_data)

## prepare stimulus data for real-task simulations
codes_data = prepare_codes_data(codes_data, stimulus_data)  

## extract true behaviors for each participant
update_codes_data_with_true_behavior(codes_data, raw_behavioral_data)
      
## run both post-hoc simulations as well as comprehensive simulations
for participant_id, info in codes_data.items():
     execute_simulations_for_participant(participant_id, codes_data)

save_file_path = 'F:/sorting_algorithm_data/generated_codes_simulation_data.pkl'
save_data_to_pickle(save_file_path, codes_data)


#execting standard algorithm simulation
save_file_path = 'F:/sorting_algorithm_data/standard_algorithms_simulation.pkl'
standard_sorting_algorithms = load_data_from_pickle(save_file_path)

standard_codes_data = codes_data.copy()
for participant_id, info in standard_codes_data.items():
     execute_standard_simmulation_for_participant(participant_id, standard_codes_data, standard_sorting_algorithms)
     
save_file_path = 'F:/sorting_algorithm_data/standard_simulation_data.pkl'
save_data_to_pickle(save_file_path, standard_codes_data)