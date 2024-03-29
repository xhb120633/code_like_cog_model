# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:09:10 2024

@author: 51027
"""

import torch
from transformers import AutoTokenizer, AutoModel
import json
import pandas as pd
import numpy as np
import re
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import sem
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.nn.functional import cosine_similarity
from numpy.random import default_rng
from multiprocessing import Pool, cpu_count
import openai

def clean_codes(tmp_code):
    # Extract the Python code block from the text
    code_cleaned_match = re.search(r"```python\n(.*?)\n```", tmp_code, re.DOTALL)
    
    if code_cleaned_match:
        code_cleaned = code_cleaned_match.group(1).strip()
        return code_cleaned
    else:
        print('unrecognized patterns in codes!')

def extract_comments_and_code(raw_code):
    """
    Extracts comments and code from a given raw code string.
    
    Parameters:
    - raw_code (str): The raw code input as a string.
    
    Returns:
    - Tuple (comments, code): Two strings containing the extracted comments and code.
    """
    # Regex pattern for single-line and multi-line comments
    pattern = r"(#.*)|('''[\s\S]*?''')|(\"\"\"[\s\S]*?\"\"\")"
    
    # Extract all comments
    comments = re.findall(pattern, raw_code)
    
    # Flatten the list of tuples and filter out empty strings
    comments = [item for sublist in comments for item in sublist if item]
    
    # Join comments into a single string, separated by new lines
    comments_str = '\n'.join(comments)
    
    # Remove comments from the code
    code_without_comments = re.sub(pattern, '', raw_code)
    
    # Remove the unified function to save space for embeddings.
    code_cleaned = remove_attempt_swap_function(code_without_comments)
    
    # Additional cleaning to remove excess whitespace and newlines
    code_cleaned = re.sub(r'\n\s*\n', '\n', code_cleaned)
    
    
    return comments_str, code_cleaned

def tokenize_nl_pl(nl_text, pl_code, include_cls=True, max_length=512):
    """
    Tokenizes natural language and programming language inputs with dynamic truncation to ensure
    the total token count does not exceed the specified maximum length.

    Parameters:
    - nl_text (str): The natural language text.
    - pl_code (str): The programming language code.
    - include_cls (bool): Whether to include the [CLS] token at the beginning.
    - max_length (int): The maximum token length including special tokens.

    Returns:
    - tokenized_ids (list): Tokenized IDs representing the truncated NL and PL inputs.
    """
    
    # Tokenize natural language and programming language texts
    nl_tokens = tokenizer.tokenize(nl_text)
    code_tokens = tokenizer.tokenize(pl_code)

    # Calculate the number of tokens to be allocated for NL and PL, considering special tokens
    num_special_tokens = 3 if include_cls else 2  # Accounting for [CLS], [SEP], [EOS]
    available_length = max_length - num_special_tokens - len(nl_tokens) - len(code_tokens)

    # If the total length exceeds the limit, truncate the code tokens first (preserving NL context)
    if available_length < 0:
        code_tokens = code_tokens[:available_length]  # Truncate code tokens to fit
        print('Limit exceeded! Compressing the code tokens!')

    # Prepare the input sequence with optional [CLS] and mandatory [SEP], [EOS] tokens
    input_sequence = [tokenizer.cls_token] if include_cls else []
    input_sequence += nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]

    # Ensure the final tokenized input does not exceed max_length
    # This additional check accounts for edge cases in token calculations
    if len(input_sequence) > max_length:
        input_sequence = input_sequence[:max_length-len([tokenizer.eos_token])] + [tokenizer.eos_token]
    
    # Convert tokens to IDs
    tokenized_ids = tokenizer.convert_tokens_to_ids(input_sequence)
    
    return tokenized_ids


def remove_attempt_swap_function(code):
    """
    Removes the specific 'attempt_swap' function from the code,
    targeting from its 'def' line through the entire function block.
    """

    pattern = r"def attempt_swap\(.*?return False, current_order\n\s*"

    # Replace the targeted block with an empty string, effectively removing it
    cleaned_code = re.sub(pattern, '', code, flags=re.DOTALL)

    return cleaned_code.strip()


    # Trim leading/trailing whitespace
    return cleaned_code.strip()


def get_text_embedding(nl_text, pl_code, include_cls=True):
    """
    Gets the text embedding for combined NL and PL texts.
    
    Parameters:
    - nl_text (str): Extracted NL text from comments.
    - pl_code (str): Extracted PL code.
    - include_cls (bool): Determines whether to use the [CLS] token's embedding or mean embedding.
    
    Returns:
    - embedding (torch.Tensor): The requested text embedding.
    """
    tokenized_ids = tokenize_nl_pl(nl_text, pl_code, include_cls=include_cls)
    inputs = torch.tensor(tokenized_ids)[None,:].to(device)

    with torch.no_grad():
        contextual_embeddings = model(inputs)[0]
    
    if include_cls:
        # Use the [CLS] token's embedding
        return contextual_embeddings[:, 0, :]
    else:
        # Use mean embedding
        return contextual_embeddings.mean(dim=1)

def update_codes_data_with_embeddings(codes_data):
    """
    Updates the `codes_data` dictionary with text embeddings for each participant.
    
    Parameters:
    - codes_data (dict): The dictionary containing participants' code data.
    """
    for participant_id, data in codes_data.items():
        if 'generated_code' in data and data['generated_code'] is not None:
            print(f"Obtaining text embeddings for codes in participant {participant_id}")
            # Use `clean_codes` if defined or directly pass raw code if not applicable
            filtered_code = clean_codes(data['generated_code'])  # Assume this function is defined
            
            # Extract comments as NL and the remaining as PL
            nl_text, pl_code = extract_comments_and_code(filtered_code)
            
            # Get tokenized IDs and then embeddings
            embedding = get_text_embedding(nl_text, pl_code, include_cls=True)  # Adjust `include_cls` as needed
            
            # Update `codes_data` with the embedding
            # Assuming you want to convert the embedding to a list for JSON compatibility or further processing
            codes_data[participant_id]['text_embeddings'] = embedding.tolist()

def encode_action(pair):
    """
    Encodes a comparison action or the 'finish' action into a unique number.
    
    Parameters:
    - pair: A tuple representing a comparison pair (item1, item2) or the string 'finish'
    
    Returns:
    - An integer representing the unique ID of the action.
    """
    if pair == 'finish':
        return 15  # Unique ID for the 'finish' action
    
    # Ensure the smallest number is always first to ignore order
    item1, item2 = sorted(pair)
    
    # Adjust formula to start with item 0
    # This formula maps pair to a unique ID based on their sorted positions, starting from 0
    id = item1 * (5 - item1 / 2) + (item2 - item1)
    
    return int(id)

def extract_correct_sequence(participant_data):
    """
    Extracts the list of action sequences from participant_data, where participant_data
    is expected to be a tuple, and one of its elements is the list of action tuples.
    Specifically looks for an element that is a list, where every item within that list
    is a tuple, indicative of action sequences.
    """
    if isinstance(participant_data, tuple):
        for element in participant_data:
            # First, ensure the element is a list
            if isinstance(element, list):
                # Next, check if all items in the list are tuples (representing actions)
                # and explicitly exclude lists that contain non-tuple items (e.g., integers, strings)
                array = []
                for item in element:
                    # Found the target list of actions
                    array.append(isinstance(item, tuple))
                if all(array):
                    return element
        # Return an empty list if no suitable action list is found within the tuple
    elif isinstance(participant_data, list):
        return participant_data
    
    else:
        return []


 

def prepare_trial_data(codes_data, trial_id, valid_participants):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sequences = []
    for pid in valid_participants:
        participant_data = codes_data[pid]['comprehensive_simulations'].get(trial_id)
        if participant_data:
            # Extract the correct sequence structure
            correct_sequence = extract_correct_sequence(participant_data)
            # Encode the actions in the sequence
            seq = [encode_action(action[0]) if isinstance(action, tuple) else encode_action(action) for action in correct_sequence]
            sequences.append(seq)

    # Determine the maximum sequence length for padding
    max_length = max(len(seq) for seq in sequences)

    padded_sequences = []
    lengths = []
    for seq in sequences:
        lengths.append(len(seq))
        pad_size = max_length - len(seq)
        padded_seq = seq + [-999] * pad_size  # Use a distinctive padding value
        padded_sequences.append(padded_seq)

    # Convert lists to tensors
    sequences_tensor = torch.tensor(padded_sequences, dtype=torch.long, device=device)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)

    return sequences_tensor, lengths_tensor

def compute_first_deviation_index(sequences_tensor):
    N, M = sequences_tensor.size()
    # Expand sequences tensor to allow element-wise comparison between all pairs
    seq_expanded_a = sequences_tensor.unsqueeze(1).expand(-1, N, -1)
    seq_expanded_b = sequences_tensor.unsqueeze(0).expand(N, -1, -1)
    
    # Compute deviation: a boolean tensor where True indicates a deviation
    deviations = seq_expanded_a != seq_expanded_b
    
    # Find the first deviation index. We use a large number to mask no-deviation cases
    first_deviation_indices = torch.where(deviations, torch.arange(M, device=sequences_tensor.device), M+1)
    min_deviation_indices = torch.min(first_deviation_indices, dim=2).values
    
    return min_deviation_indices


def compute_length_matrix(lengths_tensor):
    N = lengths_tensor.size(0)
    lengths_expanded_a = lengths_tensor.unsqueeze(1).expand(-1, N)
    lengths_expanded_b = lengths_tensor.unsqueeze(0).expand(N, -1)
    
    # Use maximum length for each pair as the denominator in similarity calculation
    length_matrix = torch.max(lengths_expanded_a, lengths_expanded_b)
    
    return length_matrix

def compute_similarity_matrix(first_deviation_indices, length_matrix):
    similarity_matrix = first_deviation_indices.float() / length_matrix.float()
    # Handle division by zero or cases with no deviation (where we set first_deviation_indices to M+1)
    similarity_matrix[first_deviation_indices > length_matrix] = 1  # Full match if no deviation within the sequence lengths
    return similarity_matrix


def compute_behavioral_similarity_matrix(codes_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    valid_participants = [pid for pid, pdata in codes_data.items() if pdata.get('generated_code') is not None]
    
    # Preprocess data for all trials and participants, convert to tensors, and pad sequences as needed
    # This is an important step that depends on your data format
    # You'll need to create tensors for each trial that contain all sequences for that trial
    
    similarity_matrix = torch.zeros((len(valid_participants), len(valid_participants)), device=device)
    
    for trial_index in range(720):  # Assuming 720 trials
        # Load or prepare the data for this trial, resulting in a tensor of shape [num_participants, sequence_length] (the dataset trial starts from 1)
        trial_sequence_data, trial_length_data = prepare_trial_data(codes_data, str(trial_index+1),valid_participants)
        trial_deviation_matrix = compute_first_deviation_index(trial_sequence_data)
        trial_length_matrix = compute_length_matrix(trial_length_data)
        trial_similarity_matrix = compute_similarity_matrix(trial_deviation_matrix, trial_length_matrix)
        similarity_matrix += trial_similarity_matrix
     
                    
    # Average the accumulated similarities
    similarity_matrix /= 720  # Normalize by the number of trials
    
    # Optionally, convert back to a CPU numpy array or another format as needed
    similarity_matrix_cpu = similarity_matrix.cpu().numpy()
    
    return similarity_matrix_cpu


def compute_code_embedding_similarities(codes_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_participants = [pid for pid, pdata in codes_data.items() if pdata.get('generated_code') is not None]
    embeddings = [codes_data[pid]['text_embeddings'][0] for pid in valid_participants]
    
    # Convert list of embeddings to a tensor and transfer to GPU
    embeddings_tensor = torch.tensor(embeddings, device=device).float()  # Ensure it's a float tensor for cosine_similarity
    
    # Compute cosine similarity matrix
    similarity_matrix_tensor = cosine_similarity(embeddings_tensor.unsqueeze(1), embeddings_tensor.unsqueeze(0), dim=2)
    
    # Convert the tensor back to a CPU numpy array
    similarity_matrix = similarity_matrix_tensor.cpu().numpy()
    
    return similarity_matrix


def compute_strategy_embedding_similarities(codes_data, behavioral_text_embeddings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract valid participants and ensure the embeddings are ordered accordingly
    valid_participants = [pid for pid, pdata in codes_data.items() if pdata.get('generated_code') is not None]
    # Assume `behavioral_text_embeddings` rows correspond to participants sorted by their IDs or as they appear in `codes_data`
    
    # Creating an index array to reorder embeddings according to `valid_participants`
    participant_indices = [int(pid) for pid in valid_participants]
    sorted_embeddings = behavioral_text_embeddings[participant_indices, 10:1547]  # Adjust indices as needed

    # Convert sorted embeddings to a PyTorch tensor and transfer to GPU
    embeddings_tensor = torch.tensor(sorted_embeddings, device=device).float()

    # Compute cosine similarity matrix in one operation
    similarity_matrix_tensor = cosine_similarity(embeddings_tensor, embeddings_tensor)
    
    # Convert the tensor back to a CPU numpy array
    similarity_matrix = similarity_matrix_tensor.cpu().numpy()
    
    return similarity_matrix


def visualize_heatmap(similarity_matrix, title, participants=None):
    # Assuming similarity_matrix is a 2D NumPy array
    matrix_size = similarity_matrix.shape[0]
    
    # Check if participants list is provided for tick labels
    if participants is None:
        # Generate default participant labels if not provided
        participants = [f'P{i}' for i in range(1, matrix_size + 1)]
    
    # Setting custom tick labels
    ticks = np.linspace(0, matrix_size - 1, num=min(matrix_size, 10), dtype=int)  # For a more general case with many participants
    tick_labels = [participants[i] for i in ticks]  # Subset of participant labels
    
    # Create the heatmap with custom ticks and labels
    plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    ax = sns.heatmap(similarity_matrix, cmap='viridis', xticklabels=tick_labels, yticklabels=tick_labels,
                     cbar_kws={"shrink": .5}, square=True)
    ax.set_xticks(ticks)  # Apply custom ticks
    ax.set_yticks(ticks)
    
    plt.title(title)
    plt.tight_layout()  # Adjust layout to not cut off edges
    plt.show()
    fig_name = 'pic/code_prediction/'+title+'.png'
    plt.savefig(fig_name,bbox_inches='tight', dpi = 600)
    
    return similarity_matrix

def flatten_and_remove_self_comparison(matrix):
    """Flatten matrix while removing self-comparison (diagonal elements)."""
    triu_indices = np.triu_indices_from(matrix, k=1)
    flattened = matrix[triu_indices]
    return flattened


def permute_and_correlate(matrix_1, matrix_2, n_permutations=2000):
    """Permute data and compute Spearman's r on CPU with parallel computation."""
    # Flatten and remove self comparisons
    flattened_1 = flatten_and_remove_self_comparison(matrix_1)
    flattened_2 = flatten_and_remove_self_comparison(matrix_2)
    true_corr, _ = spearmanr(flattened_1, flattened_2)
    
    
    results = []
    for i in range(n_permutations):
        permuted_2 = np.random.permutation(flattened_2)
        permuted_corr, _ = spearmanr(flattened_1, permuted_2)
        results.append(permuted_corr)
        if i%20 == 0:
            print(f'running {i} permutations')
    return true_corr, np.array(results)

def visualize_correlation_permutations(correlations, original_corr, title='Correlation Permutation Test'):
    """
    Visualize the results of a correlation permutation test.
    
    Parameters:
    - correlations: Array of correlation coefficients from permutation test.
    - original_corr: The original correlation coefficient computed from the non-permuted data.
    - title: Title for the plot.
    - save_path: Optional; if provided, the plot will be saved to this path.
    """
    # Calculate 95% CI of the permuted correlations
    lower_bound, upper_bound = np.percentile(correlations, [2.5, 97.5])
    
    # Plot histogram of permuted correlations
    plt.hist(correlations, bins=50, color='skyblue', alpha=0.7, label='Permuted correlations')
    
    # Mark the original correlation coefficient
    plt.axvline(original_corr, color='red', linestyle='--', label='Original correlation')
    
    # Mark the 95% CI
    plt.axvline(lower_bound, color='green', linestyle=':', label='95% CI')
    plt.axvline(upper_bound, color='green', linestyle=':', label='_nolegend_')
    
    # Add legend and labels
    plt.legend()
    plt.title(title)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    
    # Show or save the plot
    save_path = 'pic/code_prediction/'+title+'.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()
    print(f'True corr: {original_corr}; 95% CI: ({lower_bound}, {upper_bound})')


def get_codes(text_description,pseudocode):

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

    if 'Backward'  in text_description:
        text_description = 'The strategy is' + text_description + '(starting from right of the list).'
    else:
        text_description = 'The strategy is' + text_description + '(starting from left of the list).'
        
    text_description = text_description + f'The pseudocode of  this strategy in a six-length example is {pseudocode}.'
    
        
    tmp_message =  [{"role": "assistant", "content":fixed_instruction},
        {"role": "user", "content": text_description}]
    
    response = openai.ChatCompletion.create(
        model = 'gpt-4-turbo-preview',
        messages = tmp_message,
        temperature=0.1,
        max_tokens=1024,
        seed = 2024
        )

    return response

def clean_get_function(code_str, participant_id):
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

def pre_exam_for_participant(participant_id, codes_data):
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
    
    example_true_order = [1,2,3,4,5,6]
    all_permutations = list(itertools.permutations(example_true_order))
    
    attempts = 0
    max_attempts = 5
    
    ##pre-exam any potential errors in the code and send back to GPT-4 for correction
    corrected_code = code_str
    for idx, permutation in enumerate(all_permutations, start=1):
        while attempts < max_attempts:
            sort_function = clean_get_function(corrected_code, participant_id)
            
            if sort_function is None:
                print(f"Function extraction failed for participant {participant_id}.")
                break
    
            
            try:
                action_sequence = sort_function(list(permutation), example_true_order)
                if attempts > 0:
                    print("Successful correction!")
                break  # Success, exit the loop
            # except TimeoutException as t:
            #     print(f"Attempt {attempts + 1}: Infinite loop during simulation for participant {participant_id}: {t}")
            #     error = 'The code is running out of time and likely stuck in infinite loop!'
            #     corrected_code = automantic_codes_correction(description, corrected_code, error)
            #     if corrected_code is None:
            #         print("Failed to get correction from GPT-4.")
            #         return
            #     attempts += 1
            except Exception as e:
                print(f"Attempt {attempts + 1}: Error during simulation for participant {participant_id}: {e}")
                # Request correction from GPT-4
                corrected_code = automantic_codes_correction(description, corrected_code, str(e))
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


    
def execute_simulations_for_participant(participant_id, recovered_codes_data):
    """
    Executes simulations for all trials of a specific participant, using the extracted sorting function.
    Updates each trial with the simulated behavior sequence and also runs comprehensive simulations
    for all permutations of a 6-length sequence.

    Parameters:
    - participant_id: The ID of the participant.
    - participant_data: Dictionary containing all participants' data, including codes and trial info.
    """
    participant_info = recovered_codes_data.get(participant_id)
    print(f"Simulating participant {participant_id}")
    if not participant_info:
        print(f"No data found for participant {participant_id}")
        return

    code_str = participant_info.get('generated_code', '')
    sort_function = clean_get_function(code_str, participant_id)
    

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
    
    recovered_codes_data[participant_id] = participant_info

def automantic_codes_correction(text_description, original_codes, errors):

    fixed_instruction = """
Fix the Python sorting function 'human_like_sorting',which is directly executable and effective for a general sorting task, based on a strategic approach to sorting a list of items. The function should simulate the decision-making process humans might use when sorting, without direct comparison to a known 'true order'. The function must:

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

Please provide only the modified function, ready for direct execution with 'initial_stimuli' and 'true_order' inputs. Don't include any comments, explainations, notes or examples.
"""

    prompt = (
        f"The strategy description is:\n{text_description}\n\n"
        f"The original code is:\n```python\n{original_codes}\n```\n\n"
        f"The error encountered during execution is:\n{errors}\n\n"
        "Corrected code:"
    )
    
    tmp_message =  [{"role": "assistant", "content":fixed_instruction},
        {"role": "user", "content":prompt}]
    
    response = openai.ChatCompletion.create(
        model = 'gpt-4-turbo-preview',
        messages = tmp_message,
        temperature=0.1,
        max_tokens=1024,
        seed = 2024
        )
    
    corrected_codes = response['choices'][0]['message']['content']
    return corrected_codes

def attribute_strategies(participant_algorithm_similarity_df):
    attributed_strategies = {}

    for participant_id in participant_algorithm_similarity_df.index:
        # Extract the row corresponding to the participant
        similarities = participant_algorithm_similarity_df.loc[participant_id]
        
        # Check for perfect matches (similarity of 1)
        perfect_matches = similarities[similarities == 1].index.tolist()
        
        if perfect_matches:
            # If multiple perfect matches, attribute all as potential strategies
            attributed_strategies[participant_id] = perfect_matches
        else:
            attributed_strategies[participant_id] = ["Unidentified"]
    
    return attributed_strategies


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
       
openai.api_key = 'xxx' #Your Own API        
dir_name= 'C:/Users/51027/Documents/GitHub/sorting_algorithm_text_analysis/data'
file_path = 'result/generated_codes_data.json'

data_file_path = 'F:/sorting_algorithm_data/generated_codes_simulation_data.pkl'
codes_data = load_data_from_pickle(data_file_path)

behavioral_text_embeddings = np.load('data/behavioral_text_embeddings.npy', allow_pickle=True)

strategy_dic = ['Unidentified','Gnome Sort','Selection Sort','Insertion Sort','Bubble Sort','Comb Sort','Modified Selection Sort','Shaker Sort','Successive Sweeps','Backward Gnome Sort','Backward Selection Sort','Backward Insertion Sort','Backward Bubble Sort','Backward Comb Sort','Backward Modified Selection Sort','Backward Shaker Sort','Backward Successive Sweeps']



participant_data = pd.read_csv(dir_name+ '/participants.csv')
participant_data = np.array(participant_data)

#filter out the duplicate trails
participant_data = participant_data[participant_data[:,5]==False,:]
participant_data = participant_data[participant_data[:,9]==False,:]

column_names = [ 'participant_id','network_id','replication','generation', 'condition','cloned','mean_trial_score','algorithm','algorithm_description','exclusion_flag'
]
participant_data = pd.DataFrame(participant_data, columns=column_names)

### Semantic and syntax analysis for codes
## load codes-BERT for code-based text embedding
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.to(device)

##get text embeddings of codes
update_codes_data_with_embeddings(codes_data)

save_file_path = 'F:/sorting_algorithm_data/generated_codes_simulation_data.pkl'
save_data_to_pickle(save_file_path, codes_data)

##compute comprehensive simulated behavior similarity matrix
## This code has been optimized to use GPU tensor to accelerate computing, but do require a GPU memory more than 25 GB. 
## Run with A100 is recommneded.
code_simulation_behavioral_distance_matrix = compute_similarity_matrix(codes_data)

###step 1: how closely can text strategetic descriptions, code content (function logic) and behavioral outcome align?
### we need to compute similarity matrix for text embeddings and code embeddings

description_similarity_matrix = compute_strategy_embedding_similarities(codes_data, behavioral_text_embeddings)
codes_similarity_matrix = compute_code_embedding_similarities(codes_data)

##read in cloud-computed result (without running that in a causal device)
code_simulation_behavioral_distance_matrix = np.load('F:/sorting_algorithm_data/code_simulation_behavioral_distance_matrix.npy')
description_similarity_matrix = np.load('F:/sorting_algorithm_data/description_similarity_matrix.npy')
codes_similarity_matrix = np.load('F:/sorting_algorithm_data/codes_similarity_matrix.npy')

behavioral_flattened = flatten_and_remove_self_comparison(code_simulation_behavioral_distance_matrix)
text_flattened = flatten_and_remove_self_comparison(description_similarity_matrix)
code_flattened = flatten_and_remove_self_comparison(codes_similarity_matrix )

plt.hist(behavioral_flattened, alpha=0.5, label='Behavioral')
plt.hist(text_flattened, alpha=0.5, label='Text')
plt.hist(code_flattened, alpha=0.5, label='Code')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Comparative Histogram of Behavioral, Text, and Code Embeddings')
plt.legend()
save_path = 'pic/code_prediction/'+'metric_distribution'+'.png'
plt.savefig(save_path, bbox_inches='tight', dpi=600)
plt.show()


## visualize the simialrity matrix
description_similarity_array = visualize_heatmap(description_similarity_matrix, 'strategy description similarity')
codes_similarity_array = visualize_heatmap(codes_similarity_matrix, 'code similarity')
simulation_behavioral_similarity_array = visualize_heatmap(code_simulation_behavioral_distance_matrix, 'simulated behavior similarity')

#description-code alignment
original_corr1, permuted_cor_list1 = permute_and_correlate(description_similarity_array,codes_similarity_array,1)
np.save('result/permuted_cor_list1.npy', permuted_cor_list1)
#description-behavior alignment
original_corr2, permuted_cor_list2 = permute_and_correlate(description_similarity_array,code_simulation_behavioral_distance_matrix,1)
np.save('result/permuted_cor_list2.npy', permuted_cor_list2)
#code-behavior alignment
original_corr3, permuted_cor_list3 = permute_and_correlate(codes_similarity_array,code_simulation_behavioral_distance_matrix,1)
np.save('result/permuted_cor_list3.npy', permuted_cor_list3)

### load the permutation result without computing that.
permuted_cor_list1 = np.load('result/permuted_cor_list1.npy')
permuted_cor_list2 = np.load('result/permuted_cor_list2.npy')
permuted_cor_list3 = np.load('result/permuted_cor_list3.npy')

#visualize the permutation results
visualize_correlation_permutations(permuted_cor_list1, original_corr1, title='description_code_cor')
visualize_correlation_permutations(permuted_cor_list2, original_corr2, title='descrption_behavior_cor')
visualize_correlation_permutations(permuted_cor_list3, original_corr3, title='code_behavior_cor')




##conduct hierachical clustering analysis
behavioral_distance_matrix = 1 - code_simulation_behavioral_distance_matrix

behavioral_distance_matrix = squareform(behavioral_distance_matrix)
# Perform hierarchical clustering using the 'ward' method
Z = linkage(behavioral_distance_matrix, 'ward')

# Plot the dendrogram
plt.figure(figsize=(10, 8))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Participant Index')
plt.ylabel('Distance')

# Hide x-axis labels to avoid clutter
plt.xticks([])  # This hides the x-axis tick labels
save_path = 'pic/code_prediction/'+'Hierarchical Clustering Dendrogram'+'.png'
plt.savefig(save_path, bbox_inches='tight', dpi=600)
plt.show()




###Hypothesis-driven: compare with standard algorithms; use GPT-4 generate the codes and human inspect these codes
standard_sorting_algorithms = {
    "Selection Sort": {
    },
    "Gnome Sort": {     
    },
    "Shaker Sort": {
    },
    "Modified Selection Sort": {
    },
    "Bubble Sort": {
    },
    "Successive Sweeps": {
    },
    "Insertion Sort": {
    },
    "Comb Sort": {
    },
    # Adding backward versions
    "Backward Selection Sort": {
    },
    "Backward Gnome Sort": {
    },
    "Backward Shaker Sort": {
    },
    "Backward Modified Selection Sort": {
    },
    "Backward Bubble Sort": {
    },
    "Backward Successive Sweeps": {
    },
    "Backward Insertion Sort": {
    },
    "Backward Comb Sort": {
    }
}

#creating the pseudocode for each algorithm
for algorithm_name in standard_sorting_algorithms:
    standard_sorting_algorithms[algorithm_name]['pseudocode']='def\n'

for algorithm_name in standard_sorting_algorithms:
       # Retrieve the generated code for the algorithm
       pseudocode = standard_sorting_algorithms[algorithm_name]['pseudocode']
       tmp_response = get_codes(algorithm_name,pseudocode)
       generated_code = tmp_response['choices'][0]['message']['content']
       standard_sorting_algorithms[algorithm_name]['generated_code'] = generated_code
       
recovered_file_path = 'result/recovered_codes_data.json'
recovered_codes_data = load_data_from_json(recovered_file_path)      
##pre-exam the code first
for participant_id, info in standard_sorting_algorithms.items():
     pre_exam_for_participant(participant_id, standard_sorting_algorithms)

for participant_id, info in recovered_codes_data.items():
     pre_exam_for_participant(participant_id, recovered_codes_data)
## run comprehensive simulations
for participant_id, info in standard_sorting_algorithms.items():
     execute_simulations_for_participant(participant_id, standard_sorting_algorithms)
     
for participant_id, info in recovered_codes_data.items():
     execute_simulations_for_participant(participant_id, recovered_codes_data)
     
save_file_path = 'F:/sorting_algorithm_data/standard_algorithms_simulation.pkl'
save_data_to_pickle(save_file_path, standard_sorting_algorithms)

save_file_path = 'F:/sorting_algorithm_data/recovered_simulated_codes_data.pkl'
save_data_to_pickle(save_file_path, recovered_codes_data)

standard_sorting_algorithms = load_data_from_pickle(save_file_path)


load_file_path = 'F:/sorting_algorithm_data/participant_algorithm_similarity_df.csv'
participant_algorithm_similarity_df = pd.read_csv(load_file_path,index_col=0)
# Iterate over each pair of columns and check if they are identical
identical_pairs = []

for i in range(len(participant_algorithm_similarity_df.columns)):
    for j in range(i + 1, len(participant_algorithm_similarity_df.columns)):
        col1 = participant_algorithm_similarity_df.columns[i]
        col2 = participant_algorithm_similarity_df.columns[j]
        
        # Check if all values in the two columns are the same
        if participant_algorithm_similarity_df[col1].equals(participant_algorithm_similarity_df[col2]):
            identical_pairs.append((col1, col2))

# Print the pairs of columns that are identical
for pair in identical_pairs:
    print(f"Identical columns: {pair[0]} and {pair[1]}")
    

load_file_path = 'F:/sorting_algorithm_data/recovered_algorithm_similarity_df.csv'
recovered_algorithm_similarity_df = pd.read_csv(load_file_path,index_col=0)

#attributing stratgies only when similarity is 1, if none of strategy has value 1 simialrity, this should be regarded as 'Unidentified'.
attributed_strategies = attribute_strategies(participant_algorithm_similarity_df)
attributed_strategies = attribute_strategies(recovered_algorithm_similarity_df)

attributed_strategies_indexed = {pid: strategy_dic.index(strategy[0]) for pid, strategy in attributed_strategies.items()}

# Aligning participant IDs between 'participant_data' and 'attributed_strategies_indexed'
consistent_counts = [attributed_strategies_indexed[pid] == alg for pid, alg in participant_data['algorithm'].items() if pid in attributed_strategies_indexed]
consistent_counts = [strategy_dic[attributed_strategies_indexed[int(pid)]] == recovered_codes_data[pid]['true_algorithm'] for pid in recovered_codes_data if int(pid) in attributed_strategies_indexed]

# Calculate consistency rate and its standard error
mean_consistent_rate = np.mean(consistent_counts)
sem_consistent_rate = sem(consistent_counts)  # Standard error of the mean

print("Mean Consistent Rate:", mean_consistent_rate)
print("Standard Error of the Mean:", sem_consistent_rate)

# Initialize an empty matrix for the heatmap data
heatmap_data = np.zeros((len(strategy_dic), len(strategy_dic)-1))

# Loop through each strategy in 'strategy_dic'
for true_strategy_idx, true_strategy in enumerate(strategy_dic):
    # Find participants attributed with the current 'true_strategy'
    participants_attributed = participant_data[participant_data['algorithm'] == true_strategy_idx]['participant_id']
    
    # For each attributed participant, retrieve their similarity scores to all algorithms
    for predicted_strategy_idx, predicted_strategy in enumerate(strategy_dic[1:]):  # Exclude 'Unidentified' from LLM-inferred strategies
        # Collect similarities for the current 'true_strategy' against 'predicted_strategy'
        similarities = participant_algorithm_similarity_df.loc[participants_attributed, predicted_strategy].mean()
        
        # Fill the heatmap_data matrix
        heatmap_data[true_strategy_idx, predicted_strategy_idx] = similarities




# Loop through each strategy in 'strategy_dic'
for true_strategy_idx, true_strategy in enumerate(strategy_dic):
    # Find participants attributed with the current 'true_strategy' by name, converting to indices
    participants_attributed = [pid for pid, pdata in recovered_codes_data.items() if pdata['true_algorithm'] == true_strategy]
    
    for predicted_strategy_idx, predicted_strategy in enumerate(strategy_dic[1:]):
        # Collect similarities for the current 'true_strategy' against 'predicted_strategy'
        # Use list comprehension to collect similarities from participant_algorithm_similarity_df
        similarities = [recovered_algorithm_similarity_df.loc[int(pid), predicted_strategy] for pid in participants_attributed if int(pid) in recovered_algorithm_similarity_df.index]
        
        # Calculate mean similarity if there are any similarities collected
        if similarities:
            mean_similarity = np.mean(similarities)
        else:
            mean_similarity = np.nan  # Use NaN where there are no participants or similarities
        
        # Fill the heatmap_data matrix
        heatmap_data[true_strategy_idx, predicted_strategy_idx] = mean_similarity
        
# Visualize the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", xticklabels=strategy_dic[1:], yticklabels=strategy_dic, fmt=".2f")
plt.title('Averaged Similarity Between Behaviorally Assigned Strategies and LLM Inferred Strategies')
plt.xlabel('Predicted Strategies (LLM Inferred)')
plt.ylabel('True Strategies (Behaviorally Assigned)')
plt.tight_layout()
save_path = 'pic/code_prediction/'+'recovered_confusion_matrix'+'.png'
plt.savefig(save_path, bbox_inches='tight', dpi=600)
plt.show()