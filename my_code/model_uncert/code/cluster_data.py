"""
cluster_data.py

This file contains code used to read in data and cluster it based on a history of mental health data. 

Author: Ben
Date: 2024-08-25 22:44:38
"""

# Import packages
# General
import pandas as pd
from sklearn.cluster import KMeans
# This will limit the number of threads used by MKL, potentially avoiding the memory leak.
import os
os.environ['OMP_NUM_THREADS'] = '3'

def read_and_k_clust_by_hist(data_path: str, num_clusters: int, hist_start_age: int, hist_end_age: int, new_clust_col_name: str = None) -> pd.DataFrame:
    """
    This function reads in the data from a csv file and then clusters the data into the number of clusters specified.

    Args:
        data_path (str): The path to the csv file.
        num_clusters (int): The number of clusters to create.
        hist_start_age (int): The starting age for the history to be clustered on.
        hist_end_age (int): The ending age for the history to be clustered on.

    Returns:
        pd.DataFrame: The data with the cluster labels.
    """
    if new_clust_col_name is None:
        new_clust_col_name = 'hist_cluster'

    # Read
    data = pd.read_csv(data_path)
    # Filter data for clustering
    filtered_data = data[(data['age'] >= hist_start_age) & (data['age'] <= hist_end_age)]
    # Aggregate to ensure unique individual-age combinations
    filtered_data = filtered_data.groupby(['indiv_id', 'age']).agg({'mental_health': 'mean'}).reset_index()
    # fill in missing values with the mean of the column
    filtered_data = filtered_data.fillna(filtered_data.mean())
    # Pivot the data to get each individuals mental health index as a row vector
    pivoted_data = filtered_data.pivot(index='indiv_id', columns='age', values='mental_health')
    # Fill in missing values with the mean of the column
    pivoted_data = pivoted_data.fillna(pivoted_data.mean())
    # Cluster the data
    kmeans = KMeans(n_clusters=num_clusters)
    pivoted_data[new_clust_col_name] = kmeans.fit_predict(pivoted_data)
    # Merge the cluster labels back into the data
    merged_data = pd.merge(data, pivoted_data[new_clust_col_name], on='indiv_id', how='left')
    return merged_data



# run in main condition
if __name__ == "__main__":
    my_data_path = "C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/input/MH_data_for_clustering.csv"
    my_num_clusters = 2
    my_hist_start_age = 20
    my_hist_end_age = 30
    my_output = read_and_k_clust_by_hist(my_data_path, my_num_clusters, my_hist_start_age, my_hist_end_age)
    print(my_output)
    # save the output
    my_output.to_csv("C:/Users/Ben/My Drive/PhD/PhD Year 3/3rd Year Paper/Model/My Code/MH_Model/my_code/model_uncert/output/MH_data_hist_clusters.csv", index=False)
    pass