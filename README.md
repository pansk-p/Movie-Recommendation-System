# Movie Recommendation System Overview

## Introduction
This repository contains a hybrid movie recommendation system implemented in Python using collaborative filtering techniques. Collaborative filtering is a method commonly used in recommendation systems, where the system predicts a user's preferences based on the preferences of other users.

## Table of Contents
1. [Requirements](#requirements)
2. [Getting Started](#getting-started)
3. [Data](#data)
4. [Implementation Details](#implementation-details)
    - [Data Initialization](#data-initialization)
    - [Similarity Matrix Calculation](#similarity-matrix-calculation)
    - [Item-Item Collaborative Filtering (S1)](#item-item-collaborative-filtering-s1)
    - [User-Item Collaborative Filtering (S2)](#user-item-collaborative-filtering-s2)
    - [Metrics Calculation](#metrics-calculation)
5. [Results](#results)

## Requirements
Ensure you have the following Python libraries installed:

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For machine learning tools.
- `scipy`: For scientific computing functions.

## Getting Started
1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Run the main script to see the recommendation system in action.

## Data
The recommendation system uses a movie ratings dataset. The `ratings.csv` file is expected to contain information about user ratings for different movies.For this project I use the dataset from this site https://grouplens.org/datasets/movielens/latest/.

## Implementation Details

### Data Initialization
The `data_init` function reads the movie ratings data, applies preprocessing, and splits it into training and testing sets.

### Similarity Matrix Calculation
The `calculate_simMatrix` function computes the similarity matrix, allowing the user to choose between Jaccard or adjusted cosine similarity.

### Item-Item Collaborative Filtering (S1)
The `S1` function performs item-item collaborative filtering. It filters out movies with low predicted ratings, enhancing the relevance of recommendations.

### User-Item Collaborative Filtering (S2)
The `S2` function combines the filtered results from item-item collaborative filtering with user-item collaborative filtering.

### Metrics Calculation
The `calculateMetrics` function evaluates the recommendation system's performance using metrics such as Mean Absolute Error (MAE), Precision, and Recall.

## Results
Results, including MAE, Precision, and Recall, are displayed in the console.

## Usage
Run the script to train the model, make recommendations, and evaluate performance.The expected command to run the script is as follows:
```python script_name.py <source> <path> <neighbors> <train-percentage>```

   * **source:** Placeholder for the script source file.
   * **path:** Placeholder for the path to the input data file.
   * **neighbors:** Number of neighbors for similarity calculation.
   * **train-percentage:** Percentage of the dataset to be used for training.
     
For example: ```python recommendation_script.py input_data.csv 5 0.8```


## Contributing
Feel free to contribute by opening issues, proposing new features, or submitting pull requests.



