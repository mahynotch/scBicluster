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
    bicluster : CustomBiclusterClass
        Bicluster object.
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
                    permutation_num=permutation_num,
                    permutation_type='phenotype',
                    outdir=None,
                    threads=16)
    print("\nTop enriched gene sets:")
    results_df = gs_res.res2d.sort_values('NES', ascending=False).head(10)
    print(results_df.head(10))
    results_df.to_csv(os.path.join(output_dir, "top_enriched_gene_sets.csv"), index=False)

    # if not results_df.empty:
    #     top_term = results_df['Term'].iloc[0]
    #     plt.figure(figsize=(10, 6))
    #     gp.gseaplot(gs_res.results[top_term], top_term, 
    #                 ofname=os.path.join(output_dir, f"{top_term}_plot.png"))
    
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

def run_enrichr_analysis(expression_data, bicluster: CustomBiclusterClass, gene_sets='HALLMARK',
                      output_dir="enrichr_results"):
    """
    Run Gene Set Enrichment Analysis using GSEApy
    
    Parameters:
    -----------
    expression_data : DataFrame or adata
        Gene expression data with genes as rows and samples as columns.
    bicluster : CustomBiclusterClass
        Bicluster object.
    gene_sets : str or dict, optional
        Gene set database name (e.g., 'HALLMARK') or dictionary of gene sets.
    output_dir : str, optional
        Directory to store results.
    
    Returns:
    --------
    enrichr_results : dict
        Dictionary of enrichr results
    """
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    gene_list = expression_data.var_names
    up_regulated_genes = bicluster.gene_index
    down_regulated_genes = bicluster.get_negative_gene_index(np.arange(len(gene_list)))
    
    print("Running enrichr analysis...")
    enr_up = gp.enrichr(gene_list=gene_list[up_regulated_genes].astype(str).tolist(),
                    gene_sets=gene_sets,
                    outdir=None)
    enr_dw = gp.enrichr(gene_list=gene_list[down_regulated_genes].astype(str).tolist(),
                    gene_sets=gene_sets,
                    outdir=None)
    
    if "GO" in enr_up.res2d.Term.iloc[0]:
        enr_up.res2d.Term = enr_up.res2d.Term.str.split(" \(GO").str[0]
        enr_dw.res2d.Term = enr_dw.res2d.Term.str.split(" \(GO").str[0]
    enr_up.res2d['UP_DW'] = "UP"
    enr_dw.res2d['UP_DW'] = "DOWN"
    enr_res = pd.concat([enr_up.res2d.head(10), enr_dw.res2d.head(10)])
    gp.barplot(enr_res, figsize=(3,5),
                group ='UP_DW',
                title ="",
                color = ['b','r'],
                ofname=os.path.join(output_dir, "enrichr_plot.png"))
    
    enr_res.to_csv(os.path.join(output_dir, "enrichr_results.csv"), index=False)
    print("\nTop enriched gene sets:")
    print(enr_res.head(10))
    return enr_res