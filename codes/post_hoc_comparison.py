# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:11:35 2024

@author: 51027
"""
import json
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pickle
from sklearn.utils import shuffle
import random
import seaborn as sns

def compute_accuracy(codes_data):
    """
    Computes accuracies of simulated behaviors against true behaviors for each trial of each participant,
    focusing on the sequence of comparisons and padding simulated_behavior if shorter than true_behavior.

    Parameters:
    - codes_data: Dictionary containing all participants' data, including both simulated and true behaviors.

    Returns:
    - A dictionary with accuracies for each participant and trial, based on padded comparison sequences.
    """
    accuracy_results = {}

    for participant_id, participant_info in codes_data.items():
        if participant_info.get('generated_code') is None:
            continue

        participant_accuracy = {}
        trials_info = participant_info.get('real_task_simulation', {}).get('trials', {})

        for trial_id, trial_info in trials_info.items():
            simulated_behavior = trial_info.get('simulated_behavior', [])
            true_behavior = trial_info.get('true_behavior', [])

            # Correctly extract comparison sequences from both simulated and true behaviors
            simulated_comparisons = [action[0] for action in simulated_behavior if isinstance(action[0], tuple)]
            true_comparisons = [action[0] for action in true_behavior if isinstance(action[0], tuple)]


            # Pad simulated_comparisons if it's shorter than true_comparisons
            if len(simulated_comparisons) < len(true_comparisons):
                simulated_comparisons += [None] * (len(true_comparisons) - len(simulated_comparisons))

            # Calculate the accuracy as the proportion of matching comparisons
            match_count = sum(1 for sim, true in zip(simulated_comparisons, true_comparisons) if sim == true)
            accuracy = match_count / len(true_comparisons) if true_comparisons else 0
            participant_accuracy[trial_id] = accuracy

        accuracy_results[participant_id] = participant_accuracy

    return accuracy_results

def prediction_with_permutations(codes_data, num_permutations=1000):
    permuted_accuracies = []
    
    for i in range(num_permutations):
        # Shuffle codes among participants without changing the order of participants
        # Ensure the codes_data dictionary doesn't get modified directly by working on a copy for each permutation
        permuted_codes = shuffle([data['generated_code'] for pid, data in codes_data.items() if 'generated_code' in data])
        
        # Create a copy of codes_data to avoid modifying the original during permutations
        permuted_data = {pid: data.copy() for pid, data in codes_data.items()}
        
        # Assign permuted codes back to participants in the copied data
        for (pid, data), code in zip(permuted_data.items(), permuted_codes):
            if 'generated_code' in data:
                data['generated_code'] = code
        
        #run simulation for permuted codes
        for participant_id, info in permuted_data.items():
             simulating_permuted_data(participant_id, permuted_data)
             
        #compute raw accuracy on each trial
        accuracy_data = compute_accuracy(codes_data)

        # For each permutation, calculate and store the mean accuracy
        subject_wise_accuracy = {participant: np.mean(list(trials.values())) for participant, trials in accuracy_data.items()}

        # Extract accuracies for plotting
        permuted_mean_accuracy = np.mean(list(subject_wise_accuracy.values()))
        
        permuted_accuracies.append(permuted_mean_accuracy)
        
        if i%20 == 0:
            print(f'completing {i} permutations')
    
    # Return the list of permuted accuracies without computing the original accuracy or p-value
    return permuted_accuracies

def simulating_permuted_data(participant_id, permuted_data):
    """
    Executes simulations for all trials of a specific participant, using the extracted sorting function.
    Updates each trial with the simulated behavior sequence and also runs comprehensive simulations
    for all permutations of a 6-length sequence.

    Parameters:
    - participant_id: The ID of the participant.
    - participant_data: Dictionary containing all participants' data, including codes and trial info.
    """
    participant_info = permuted_data.get(participant_id)
    #print(f"Simulating participant {participant_id}")
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

def visualize_accuracy_permutations(correlations, original_corr, title='Post-hoc Accuracy Permutation Test'):
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
    plt.hist(correlations, bins=50, color='skyblue', alpha=0.7, label='Permuted accuracies')
    
    # Mark the original correlation coefficient
    plt.axvline(original_corr, color='red', linestyle='--', label='Original accuracy')
    
    # Mark the 95% CI
    plt.axvline(lower_bound, color='green', linestyle=':', label='95% CI')
    plt.axvline(upper_bound, color='green', linestyle=':', label='_nolegend_')
    
    # Add legend and labels
    plt.legend()
    plt.title(title)
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    
    # Show or save the plot
    save_path = 'pic/code_prediction/'+title+'.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.show()
    return((lower_bound,upper_bound))
            
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
column_names = ["participant_id","network_id","generation","condition","mean_trial_score","cloned","replication","trial_index","state","image_index","image_rank"]
stimulus_data = pd.DataFrame(stimulus_data, columns = column_names)

summary_behavioral_data = pd.read_csv(dir_name+ '/trials.csv')
summary_behavioral_data = np.array(summary_behavioral_data )

raw_behavioral_data = pd.read_csv(dir_name+ '/comparisons.csv')
column_names =['participant_id','network_id','generation','condition','mean_trial_score','cloned','replication','trial_index','comparison_index','image_i_position_index','image_j_position_index','swap','rank_image_i','rank_image_j']
raw_behavioral_data = pd.DataFrame(raw_behavioral_data, columns = column_names)
raw_behavioral_data = raw_behavioral_data.loc[raw_behavioral_data['cloned'] == False]

file_path = 'result/generated_codes_data.json'

data_file_path = 'F:/sorting_algorithm_data/generated_codes_simulation_data.pkl'
codes_data = load_data_from_pickle(data_file_path)
       
#compyte post-hoc accuracy
accuracy_data = compute_accuracy(codes_data)

# inspect those data which may have problems in comparisons (check complete zero participants)
zero_accuracy_participants = []
for participant_id, tmp_accuracies in accuracy_data.items():
    if all(tmp_accuracies.values() == 0):
        zero_accuracy_participants.append(participant_id)
        
##exclude practice trials
subject_wise_accuracy = {participant: np.mean(list(trials.values())[3:]) for participant, trials in accuracy_data.items() if len(trials.values()) > 3}

# Assuming subject_wise_accuracy is already computed and available
# Sort participants based on their accuracies and select the top 10
top_participants = sorted(subject_wise_accuracy.items(), key=lambda x: x[1], reverse=True)[:10]

# Print the top 10 participant IDs with their accuracies
for participant, accuracy in top_participants:
    print(f"Participant ID: {participant}, Accuracy: {accuracy}")

# Convert subject_wise_accuracy to DataFrame
accuracy_df = pd.DataFrame.from_dict(subject_wise_accuracy, orient='index', columns=['accuracy'])

# Reset index to make 'participant_id' a column
accuracy_df.reset_index(inplace=True)
accuracy_df.rename(columns={'index': 'participant_id'}, inplace=True)

# Ensure participant_id columns are of the same type
participant_data['participant_id'] = participant_data['participant_id'].astype(str)
accuracy_df['participant_id'] = accuracy_df['participant_id'].astype(str)

# Merge accuracy_df with participant_data
merged_data = pd.merge(participant_data, accuracy_df, on='participant_id')

# Filter for algorithm == 1
algorithm_1_data = merged_data[merged_data['algorithm'] == 10]

# Sort by accuracy in descending order
sorted_algorithm_1_data = algorithm_1_data.sort_values(by='accuracy', ascending=False)

# Extract the top 10 participant ids
top_10_participants = sorted_algorithm_1_data.head(10)['participant_id'].values

print("Top 10 best predicted participant ids for algorithm 1:", top_10_participants)


# Extract accuracies for plotting
accuracies = list(subject_wise_accuracy.values())

###run additional permutation analysis
###use the uncimulated codes_data and permutes codes-participant correspondence to simulate the data
file_path = 'result/generated_codes_data.json'

codes_data = load_data_from_json(file_path)

permuted_accuries = prediction_with_permutations(codes_data, 2000)

np.save('result/permuted_accuracy.npy', permuted_accuries)

## Directly load the permutation result instead of running it.
permuted_accuries = np.load('result/permuted_accuracy.npy')
permuted_CI = visualize_accuracy_permutations(permuted_accuries, np.mean(accuracies))


# Plotting Figure 3
plt.figure(figsize=(6, 4))
plt.hist(accuracies, bins=50, color='#FDDEBE', edgecolor='black', label='Participant Accuracies')
plt.title('Subject-wise Accuracy Distribution')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')

# Random chance level
random_chance = 1/16
plt.axvline(random_chance, color='skyblue', linestyle='--', label='Random Chance (1/16)')
plt.axvline(np.mean(accuracies), color='red', linestyle='--', label='Mean Accuracy')
# 95% CI of null distribution
plt.axvline(permuted_CI[0], color='green', linestyle='--', label='Prior Knowledge 95% CI')
plt.axvline(permuted_CI[1], color='green', linestyle='--', label= None)

# Calculate the proportion of participants surpassing the chance level and the upper bound of CI
proportion_above_chance = np.mean(np.array(accuracies) > random_chance)
proportion_above_upper_CI = np.mean(np.array(accuracies) > permuted_CI[1])

plt.legend()
plt.show()
plt.savefig('pic/code_prediction/individual_accuracy_distribution.png', bbox_inches='tight', dpi=600)

# Returning the proportions
print(f"Proportion of participants surpassing random chance: {proportion_above_chance:.2f}")
print(f"Proportion of participants surpassing the 95% CI upper bound: {proportion_above_upper_CI:.2f}")


#### is a better performed participant more predictable from their strategtic description?
# Extract mean_trial_scores for participants existing in accuracy_data (for this correlation, the accuracy should remove the first 3 trials, since the task performance is measured starting the 4th trial)
participant_data_df = pd.DataFrame(participant_data).set_index('participant_id')
participant_data_df.index = participant_data_df.index.astype(str)

# Extract mean_trial_scores for participants existing in accuracy_data
mean_trial_scores = []

for pid in accuracy_data.keys():
    if int(pid) in participant_data_df.index:
        mean_trial_scores.append(participant_data_df.at[int(pid), 'mean_trial_score'])




# Prepare a DataFrame from accuracy_data for easier manipulation
accuracy_df = pd.DataFrame(list(subject_wise_accuracy.items()), columns=['participant_id', 'accuracy']).set_index('participant_id')

# Merge accuracy data with participant data to associate each accuracy score with an algorithm
merged_data = participant_data_df.merge(accuracy_df, left_index=True, right_index=True)

# Iterate through each algorithm in strategy_dic to plot
# You can get overall plot directly calling algorithm_data = merge_data (without specifying the algorithm)
for i, algorithm_name in enumerate(strategy_dic):
    # Filter data for the current algorithm
    algorithm_data = merged_data[merged_data['algorithm'] == i]

    # Continue only if we have data for this algorithm
    if not algorithm_data.empty:
        x = algorithm_data['mean_trial_score']
        x = pd.to_numeric(x, errors='coerce')

        y = algorithm_data['accuracy']

        # Create a joint plot with by-side density plots and scatter plot with regression line
        jointplot = sns.jointplot(x=x, y=y, kind='reg', color='#FDDEBE', joint_kws={'scatter_kws': dict(alpha=0.5)}, line_kws={'color': 'red'})
        
        # Adjust the size of the plot
        jointplot.fig.set_figwidth(6)  # Set the width of the figure
        jointplot.fig.set_figheight(4)  # Set the height of the figure
        
        # Enhancements
        jointplot.fig.suptitle(f'{algorithm_name}', fontsize=10, y=1.00)

        # Adjust subplot parameters to give more space to the title
        jointplot.fig.subplots_adjust(top=0.95)
        jointplot.set_axis_labels('Mean Trial Score', 'Prediction Accuracy')

        # Adjust ticks and limits
        ticks = np.linspace(0, 1, num=5)
        jointplot.ax_joint.set_xlim(0, 1)
        jointplot.ax_joint.set_ylim(0, 1)
        jointplot.ax_joint.set_xticks(ticks)
        jointplot.ax_joint.set_yticks(ticks)

        # Save the plot with a unique name
        plt.savefig(f'pic/code_prediction/performance_accuracy_corr/performance_accuracy_{algorithm_name}.png', dpi=600)
        plt.close()  # Close the figure to avoid displaying it inline if running in a notebook


##from a time perspective, is a later trial would yield higher accuracy (participants are more familiarzied and the strategtic description may more align with the recent behavior)
# Convert accuracy_data to a format suitable for analysis by trial
trial_accuracies = {str(trial): [] for trial in range(13)}  

for participant, trials in accuracy_data.items():
    for trial, accuracy in trials.items():
        if trial in trial_accuracies:
            trial_accuracies[trial].append(accuracy)


# Calculate mean accuracy and standard error for each trial
mean_accuracies = [np.mean(trial_accuracies[trial]) for trial in trial_accuracies]
std_errors = [np.std(trial_accuracies[trial]) / np.sqrt(len(trial_accuracies[trial])) for trial in trial_accuracies]

# Plotting
trial_labels = list(range(1, 14))
plt.figure(figsize=(6, 4))
plt.bar(trial_labels, mean_accuracies, yerr=std_errors, capsize=5, color='#FDDEBE', edgecolor='black')
plt.xlabel('Trial')
plt.ylabel('Accuracy')
plt.title('Accuracy Across Trials')
plt.xticks(trial_labels)


# Correlational analysis between trial number and mean accuracy
trial_numbers = list(range(1, 14))
correlation_coef, p_value = pearsonr(trial_numbers, mean_accuracies)
# plt.figtext(0.15, 0.85, f'Pearson Correlation: {correlation_coef:.2f}\nP-value: {p_value:.2e}')
plt.show()
plt.savefig('pic/code_prediction/performance_trial_effect.png',bbox_inches='tight', dpi = 600)
correlation_coef, p_value

##report this filtered accuracy
np.mean(accuracies)

## is word count correlate with the performance?
word_count = []
for pid in subject_wise_accuracy:
    word_count.append(len(codes_data[pid]['algorithm_description']))

x = word_count
y = accuracies
# Set the figure size
plt.figure(figsize=(6, 4))

# Plot scatter and regression line with confidence interval
sns.regplot(x=x, y=y, color='#FDDEBE', scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

# Enhancements
plt.title('word count vs. prediction performance')
plt.xlabel('Word Count')
plt.ylabel('Prediction Accuracy')

# Adjust ticks and limits
ticks = np.linspace(0, 1, num=5)
plt.ylim(0, 1)
plt.yticks(ticks)

# Save the plot with a unique name
plt.savefig('pic/code_prediction/word_count_prediction_accuracy.png', bbox_inches='tight', dpi=600)
plt.show()