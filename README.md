# Reproducibility Materials for "From Strategic Narratives to Code-Like Cognitive Models: An LLM-Based Approach in a Sorting Task"

This repository contains the materials necessary for reproducing the results presented in our paper, which serves as an extension of the work by Thompson et al., 2022. The paper introduces a novel approach to analyzing strategic narratives and generating code-like cognitive models to mimic human-like sorting behaviors using Large Language Models (LLMs), specifically GPT-4. This work has been accepted by __[The 1st Proceedings of the Conference on Language Modeling (COLM)](https://colmweb.org/index.html)__!

## Overview

This paper illustrates how large language models can turn verbal reports into programming codes, serving as cognitive models to predict and interpret human behaviors in a sorting task. This README provides an overview of each script and its role in the research process, as well as information on how to replicate our findings and use our data.

### Scripts and Their Functions

- **strategy_text_analysis.py**: Analyzes text embeddings and represents and predicts behaviorally discovered strategies.
- **code_generation.py**: Generates programming codes as cognitive models for human-like sorting behaviors by calling the GPT-4 API.
- **executing_codes.py**: Extracts, debugs, and simulates codes for posthoc analysis (to compare with human data) and comprehensive simulations (to evaluate how codes are algorithmically alike from each other and to compare to standard algorithms).
- **post_hoc_comparison.py**: Evaluate how well the generated code-like models predict true human behaviors and captures human-related effects.
- **program_analysis.py**: Analyzes the representation of descriptions, codes, and simulated behaviors. It also attributes generated code-like models to standard algorithms.
- **LLM_generated_description.py**: Generates strategy descriptions and the codes based on these descriptions for each standard algorithm. It is used for a recovery test to evaluate the self-consistency of the LLM in understanding sorting algorithms.
- **algorithm_prediction.py**: Predict standard algorithms from the strategy descriptions by several LLMs and compare the predictive performance of the closet standard algorithms (predicted before) and generated codes.

### Data Files

- **codes_data.json** and **recovered_codes_data.json**: Located in the result folder, these files contain the generated codes and the recovered codes, respectively. Parameter settings and prompts are included, but you will need your own API key to regenerate codes from scratch.

### Replicating Our Results

To replicate our results, follow the steps outlined by the scripts mentioned above. Note that simulation results are not included due to their size, but they can be generated by running `executing_codes.py`, which should complete within 2-5 minutes. For efficiency, the similarity matrix result has been optimized for GPU implementation.

### Additional Notes

All permutation results are kept with full samples. Please be aware that to replicate the study from the beginning, you will need to use your own API key for the GPT-4 calls.







# Below is the Readme content from the original paper about the raw data strcture. Sources from [Thompson et.al., 2022](https://www.science.org/doi/10.1126/science.abn0915)

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
