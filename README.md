# Reproducibility materials for `Complex cognitive algorithms preserved by selective social learning in experimental populations`


# Call For Papers: 1st Conference on Language Modeling (COLM) 2024

## Key Dates

- **Abstract Submission Deadline**: March 22, 2024 (Anywhere on Earth)
- **Full Paper Submission Deadline**: March 29, 2024 (Anywhere on Earth)
- **Notification of Acceptance**: July 9, 2024
- **Conference Dates**: October 7-9, 2024

# Strategy-Based Model Creation and Evaluation

## Overview
The goal is to create models that simulate participants' behavior in a sorting task, using their natural language strategy descriptions. These models should capture the essence of participants' decision-making processes and allow for evaluation against actual behaviors and comparison among different models.

## Model Creation

### Step 1: Strategy Abstraction
- **Extract Strategy**: Identify common algorithms or approaches from participants' descriptions.
- **Algorithm Abstraction**: Generalize these strategies to be adaptable and not instance-specific.

### Step 2: Model Structure
- **Memory and Internal Beliefs**: Implement data structures to track the believed order of pictures and the outcomes of previous comparisons.
- **General Functionality**:
  - **Update Beliefs**: Modify internal beliefs based on feedback.
  - **Decide Next Action**: Use the abstracted strategy to choose the next pair of pictures.

### Step 3: Program as a Function
- **Input**: The current state and history of actions.
- **Process**: Apply strategy to decide the next action.
- **Output**: The next pair of pictures to be compared.

## Evaluation

### Evaluating Accuracy
- **Direct Comparison**: Compare the program's generated action sequence with the participant's actual sequence.

## Programs Analysis

### Semantic and Sytatic Analysis

#### CodeBERT Embeddings (Code2vec)
- Use a code-based embedding model to convert code content to text embeddings

#### Abstract Syntax Trees(ASTs）
- Use AST to analyze the static structure of the program.

### Behavioral Analysis

#### Simulation Across Conditions
- Run each program across all possible permutations of pictures to generate a comprehensive behavioral profile.

#### Behavioral Clustering
- Apply clustering algorithms to group programs based on the similarity in decision-making patterns.

### Further Analysis

#### Cluster Analysis
- Investigate common strategies within clusters to understand algorithmic similarities.

#### Inter-cluster Comparison
- Compare different clusters to identify unique or innovative strategies.

#### Algorithmic Identity
- Investigate if programs produce similar behaviors across different numbers of pictures, suggesting identical underlying algorithms.

## Optimizing Generated codes (a better fit for behaviors)
### In-context Learning
- Provide sample data in prompt
### Fine-tune (what are the golden codes?)

### Fitting free parameters to capture noises in human behavior

### Using Neural Networks to discover codes?

### Bayesian program induction to *fit* data for the best performance in code space.

## Technical Implementation

1. **Simulate Behaviors**: Generate and record actions for each program across all permutations.
2. **Behavioral Data Representation**: Format the decision-making patterns for clustering.
3. **Apply Clustering Algorithms**: Group programs based on behavioral similarity.
4. **Analysis and Visualization**: Use techniques like MDS or t-SNE for visualization and further analysis.

This approach allows for a detailed understanding of the strategies used by participants, how well models replicate these strategies, and the algorithmic relationships between different models.


## Experiment codebase
The codebase we used to conduct the experiment is included in the `experiment/` directory. This codebase relies on [Dallinger](https://github.com/Dallinger/Dallinger).

## Data
The results of the multi-generational experiment are provided in the datafiles included in the `data/` directory and summarised below. The `data/exploratory-transmission-modality-study` subdirectory contains results from of the exploratory follow-up study examining transmission modality, formatted using an analagous strcuture (see below).

### Datafile: `data/participants.csv`
Participant-level information. One row per participant.

* `participant_id`: participant-level identifier, consistent across all datafiles.
* `network_id`: identifier for the specific network that a participant belonged to.
* `replication`: an identifier for tracking pairs of networks that were yoked to a shared initial generation; we thought of each pair of yoked networks as a single replication of the evolutionary process, so for example networks 1 and 11 were yoked to the same initial generation and will both be identified as `replication` 1, networks 2 and 12 as `replication` 2, etc.
* `generation`: generation number.
* `condition`: which treatment the participant was assigned to (Asocial = A, Random Mixing = RM, Selective Social Learning = SSL).
* `cloned`:  **Note:** half of the `generation = 0` participant rows in this dataset are exact duplicates of the other half of the `generation = 0` rows; we included these duplicate rows because they simplify several visualization algorithms; they are relevant because the Asocial initial `generation` (0) in any `replication` is half the sample size of the other generations, because both `condition`s share an initial generation; as a result, rows where `clonded = True` will duplicate a `participant_id` but will have a different `network_id` and `condition` than their counterpart. Remove `cloned = True` rows for most analysis purposes.   
* `mean_trial_score`: average task performance for this participant calculated across 10 scored trials and scaled to lie in 0 (no sucessful trials) to 1 (maximum achievable under the objective function)
* `algorithm`: numerical index of the algorithm assigned to this participant by the machine learning analysis; ranges 0-16, where 0 = no identifiable algorithm, 1-8 identifies the attested algorithm classes executed "forwards", and 9-16 identifies the algorithm classes executed in reverse (see below); 
* `algorithm_description`: the verbal description left by the participant describing their strategy for use by future participants.
* `exclusion_flag`: over-recruited and non-completing participant have already been removed from this dataset, but the dataset does still include a small proportion of participants we suspect either violated repeat-participation rules through duplicate mechanical turk worker accounts or were not engaging with the task. The `exclusion_flag = True` identifies these participants.   

##### Algorithm identifiers

Here are the names of the attested algorithms (and their reversed-execution counterparts) and their numerical identifiers:

* `0`: Unidentified
* `1`: Gnome sort
* `2`: Selection sort
* `3`: Insertion sort
* `4`: Bubble sort
* `5`: Comb sort
* `6`: Modified Selection sort
* `7`: Shaker sort
* `8`: Successive sweeps
* `9`: Gnome sort (Rev.)
* `10`: Selection sort (Rev.)
* `11`: Insertion sort (Rev.)
* `12`: Bubble sort (Rev.)
* `13`: Comb sort (Rev.)
* `14`: Modified Selection sort (Rev.)
* `15`: Shaker sort (Rev.)
* `16`: Successive sweeps (Rev.)

In most of our analyses we collapsed the forwards and reversed executions of an algorithm into a single algorithm class. For example, here is the python snippet illustrating this transformation on a Pandas dataframe created from the `participants.csv` dataset: 

```
participants['algorithm'] = (
    participants
    .algorithm
    .apply(lambda a: a if a <= 8 else a - 8)
    .astype(int)
)

```

### Datafile: `data/comparisons.csv`

Comparison-level information. Each row describes a single comparison made by a specific participant during a specific trial. 

* `participant_id`: participant-level identifier, consistent across datafiles.
* `network_id`: network identifier, consistent across datafiles.
* `generation`: generation number.
* `condition`: which treatment the participant was assigned to (Asocial = A, Random Mixing = RM, Selective Social Learning = SSL).
* `mean_trial_score`: average task performance for this participant calculated across 10 scored trials and scaled to lie in 0 (no sucessful trials) to 1 (maximum achievable under the objective function)
* `cloned`: see above
* `replication`: see above
* `trial_index`: numerical index of the trial that this comparison was made in; 0 - 2 are task familiarisation trials; 3 - 12 are scored trials
* `comparison_index`: numerical index of the comparison within this trial (trial index resets to zero at the beginning of each trial)
* `image_i_position_index`: an index identifying the position in the array (from left to right) of the first stimulus selected during this comparison
* `image_j_position_index`: an index identifying the position in the array (from left to right) of the second stimulus selected during this comparison
* `swap`: True if the images compared were out of rank order and therefore swapped position, False otherwise
* `rank_image_i`: the rank (hidden order from 1 to 6) of the first stimulus selected during this comparison
* `rank_image_j`: the rank (hidden order from 1 to 6) of the first stimulus selected during this comparison

### Datafile: `data/networks.csv`
A representation of the network structure implied by social learning in the experiment. Each row in this dataset records a participant at generation t + 1 choosing to observe the description and demonstration provided by a participant at generation t. Participants could select up to three unique demonstrators from the previous generation and could observe descriptions and demonstrations more than once.  

* `network_id`: network identifier, consistent across datafiles.
* `participant_id`: participant-level identifier for the participant making the observation.
* `parent_participant_id`: participant-level identifier for the participant being observed
* `parent_rank`: this field records the order of observation. i.e. the smallest rank among demonstrators selected by a specific participant was the first demonstrator observed by that participant; the second smallest rank among demonstrators selected by a specific participant was the second observation made by the participant, etc/


### Datafile: `data/orderings.csv`
Stimulus-level information recording the initial and final orderings of each stimulus in the presentation array for a given trial for a given participant.

* `participant_id`: participant-level identifier, consistent across datafiles.
* `network_id`: network identifier, consistent across datafiles.
* `generation`: generation number.
* `condition`: which treatment the participant was assigned to (Asocial = A, Random Mixing = RM, Selective Social Learning = SSL).
* `mean_trial_score`: average task performance for this participant calculated across 10 scored trials and scaled to lie in 0 (no sucessful trials) to 1 (maximum achievable under the objective function)
* `cloned`: see above
* `replication`: see above
* `trial_index`: numerical index of the trial that this comparison was made in; 0 - 2 are task familiarisation trials; 3 - 12 are scored trials
* `state`: this variable indicates whether the record describes a stimulus in the `initial` (hidden) or `final` ordering for this trial
* `image_index`: an index identifying the position in the array (from left to right) of the stimulus
* `image_rank`: the rank (hidden order from 1 to 6) of the stimulus

### Datafile: `data/trials.csv`
Trial-level information recording outcome and taask performance. Each row describes a single trial by a specific participant.

* `participant_id`: participant-level identifier, consistent across datafiles.
* `network_id`: network identifier, consistent across datafiles.
* `generation`: generation number.
* `condition`: which treatment the participant was assigned to (Asocial = A, Random Mixing = RM, Selective Social Learning = SSL).
* `cloned`: see above
* `replication`: see above
* `trial_index`:  numerical index of the trial that this comparison was made in; 0 - 2 are task familiarisation trials; 3 - 12 are scored trials
* `num_comparisons`: number of comparisons made by the participant during this trial
* `trial_successful`: indicates whether the trial was succesful (the final ordering was the correct hidden ordering)
* `trial_score`: task performance on this trial scaled to lie in 0 (no sucessful trials) to 1 (maximum achievable under the objective function)

## Analysis
Python notebooks containing the codebase used for our analyses are included in the `analysis` directory. While there will likely be analyses that these notebooks do not cover in full by the point of publication, we include these notebooks in an effort to provide transparency into our analysis pipeline and reproducible examples of the core analyses that support the primary findings presented in the paper. 

The `performance-analysis` subdirectory contains notebooks implementing the regression analyses underpinning our findings surrounding task performance rates. The `algorithm-analysis` subdirectory contains notebooks implementing the regression analyses underpinning our findings surrounding the discovery and transmission of particular algorithms, and selective social learning.
