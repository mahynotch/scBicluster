from gseapy import Biomart
bm = Biomart()
import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
from .utils import extract_mat, CustomBiclusterClass
import os, anndata

def run_gsea_analysis(expression_data, gene_sets='HALLMARK', class_vector=None,
                      output_dir="gsea_results", permutation_num=1000):
    """
    Run Gene Set Enrichment Analysis using GSEApy
    
    Parameters:
    -----------
    expression_data : DataFrame or adata
        Gene expression data with genes as rows and samples as columns.
    gene_sets : str or dict, optional
        Gene set database name (e.g., 'HALLMARK') or dictionary of gene sets.
    class_vector : list or str, optional
        Class labels for samples.
    output_dir : str, optional
        Directory to store results.
    permutation_num : int, optional
        Number of permutations. For real analysis, 1000 is recommended.
    
    Returns:
    --------
    gsea_results : dict
        Dictionary of GSEA results
    """
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if isinstance(expression_data, anndata.AnnData):
        expr_data = expression_data.to_df().T
    elif isinstance(expression_data, pd.DataFrame):
        expr_data = expression_data
    else:
        raise ValueError("expression_data must be an AnnData or DataFrame object")
        
    if class_vector is None:
        raise ValueError("When providing expression data, class_vector must also be specified")
    
    print("Running GSEA analysis...")
    gs_res = gp.gsea(data=expr_data,
                    gene_sets=gene_sets,
                    cls=class_vector,
                    method='phenotype', 
                    permutation_num=permutation_num,
                    outdir=output_dir)
    
    print("\nTop enriched gene sets:")
    results_df = gs_res.res2d.sort_values('NES', ascending=False).head(10)
    print(results_df[['Term', 'ES', 'NES', 'pval', 'fdr']])
    
    if not results_df.empty:
        top_term = results_df['Term'].iloc[0]
        plt.figure(figsize=(10, 6))
        gp.gseaplot(gs_res.results[top_term], top_term, 
                    ofname=os.path.join(output_dir, f"{top_term}_plot.png"))
    
    return gs_res.results

def bicluster_gesa(adata, bicluster: CustomBiclusterClass, gene_sets='HALLMARK', 
                      output_dir="gsea_results", permutation_num=1000):
    adata.copy()
    class_vector = np.zeros(adata.shape[0])
    class_vector[bicluster.cell_index] = 1
    class_vector = pd.Categorical(class_vector)
    class_vector = class_vector.rename_categories({0: "Other", 1: "Bicluster"})
    run_gsea_analysis(adata, gene_sets=gene_sets, class_vector=class_vector,
                      output_dir=output_dir, permutation_num=permutation_num)
    