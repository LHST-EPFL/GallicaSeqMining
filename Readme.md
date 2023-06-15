# Project Name : Identifying User Navigation Archetypes In Gallica navigation logs: Sequence Mining and Clustering based approach
Mohamed Aziz Ben Chaabane : 257884
Ahmed Nour Achiche : 263047

Supervisors : 
- Jérôme Baudry
- Simon Dumas Primbault

2022-2023

## About :
This project is aimed at understanding and documenting user behaviors on online platforms, especially the digital library, Gallica. The primary focus is to identify and analyze digital footprints, namely server log data, using advanced data processing algorithms and sequence mining techniques to reveal key user paths and online research strategies. The extracted sequences are then grouped into distinct categories using clustering algorithms. Potential future investigations may include a comparative study of navigation paths across different research domains. This study presents a profound intersection of data analysis and sociology, aimed at uncovering technological and sociological patterns.

## Research summary :
The project presents the work conducted in filtering and processing user requests to
extract meaningful actions and develop user sessions. Our effort focused on creating sequences of
actions to better understand user behaviour.
Three different methods were explored for the task: LSTM, SGT, and SPM. LSTM was found to be
unsuitable due to its high sensitivity to sequence length. SGT performed well on smaller datasets but
led to an excessive number of clusters when applied to larger data. SPM emerged as the most effective,
scaling well and resulting in eight behaviours that could be related to typical user behaviour
![No image](.graphs/pattern_features_clustering/action_distribution.png)
## Instructions

This guide provides instructions for running the project's scripts to download, preprocess, and analyze data. Follow the steps below to get started.

### Prerequisites

- Python 3.x
- Additional libraries mentioned in the requirements.txt file

### Step 1: Download Data

1. Open the `from_NAS_to_cluster.py` file.
2. Locate the dedicated place for the username and password.
3. Replace `<username>` and `<password>` with your actual credentials.
4. Run the `from_NAS_to_cluster.py` script to download the data from the lab NAS to the cluster storage.

### Step 2: Preprocess Data

1. Run the `process_chunks.py` script to preprocess the downloaded data.
   ```
   python process_chunks.py
   ```

### Step 3: Form User Sessions

1. Run the `form_sessions_from_chunks.py` script to form user sessions.
   ```
   python form_sessions_from_chunks.py
   ```

### Step 4: Collate Session Information

1. Run the `collate_sessions.py` script to join all session information required for our methods into a single file.
   ```
   python collate_sessions.py
   ```
### Step 5: Run the Methods

open notebook `cluster_spm.ipynb` to run the SPM method or `cluster_sgt.ipynb` to run the SGT method.


Note: Make sure to review the code and modify any other necessary parameters or configurations based on your specific setup and requirements.

## License

This project is licensed under the [MIT License](./LICENSE).