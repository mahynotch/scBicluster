# scBicluster
scBicluster is a tool designed for calling and evaluating biclustering algorithms in single-cell RNA sequencing data. It provides a unified interface for various biclustering methods and facilitates the comparison of their performance on real and synthetic datasets. It supports interacting with scanpy as well. The tools offer the following functions.

- Wrapper for a variety of biclustering algorithms, including spectral biclustering, scBC, ISA, LAS, QUBIC, CC, Bibit, etc. and also a random bicluster generator for performance evaluation.
- Evaluation metrics for assessing the quality of biclusters, such as consensus score, jaccard index, wilcoxon rank-sum test, and recovery score.
- Visualization tools for exploring the structure of biclusters and their relationships to other data, including umap with convex hulls and clustered heatmap.
- Enrichment test for identifying biological pathways or gene sets that are overrepresented (or underrepresented) in the discovered biclusters.