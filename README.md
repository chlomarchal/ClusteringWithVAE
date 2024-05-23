# Clustering with Variational Autoencoders :

This code is a support for a Master Thesis. 

Writer = Chloé Marchal

The goal of the thesis is to reduce the dimensionality of a dataset using a VAE and perform clustering on the mapped dataset, as well as regression. 
Additionally, we try to generate some synthetic data and again perform clustering on it. 

Here is a short explanation of the files : 
- clustering_burt.py : code for encoding the dataset using the Burt matrix, and clustering that encoded dataset. 
- data_generation.py : code for the data generation. 
- descriptive_analysis.py : code for all the descriptive plots.
- fine-tuning_and_graphs.py : code for finding the hyperparameters for the VAE for performing regression and clustering and for most graphs in the case study.
- vae_kmeans_glm.py : code for the dimensionality reduction using the VAE and performing clustering and regression. 
