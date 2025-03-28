import numpy as np
from typing import List
from .algorithms import cliff_delta_bicluster, one_side_mannwhitneyu_bicluster, compute_coverage
from .visualization import plot_biclusters_umap, plot_biclusters_heatmap
import pandas as pd
from scipy import stats, sparse

def bicluster_jaccard(bicluster1, bicluster2) -> float:
    # bicluster1 and bicluster2 are tuples of (rows, columns)
    rows1, cols1 = bicluster1.cell_index, bicluster1.gene_index
    rows2, cols2 = bicluster2.cell_index, bicluster2.gene_index
    
    # Calculate intersection and union of rows and columns
    rows_intersection = len(np.intersect1d(rows1, rows2))
    cols_intersection = len(np.intersect1d(cols1, cols2))
    rows_union = len(np.union1d(rows1, rows2))
    cols_union = len(np.union1d(cols1, cols2))
    
    # Calculate Jaccard index
    if rows_union == 0 or cols_union == 0:
        return 0
    
    similarity = (rows_intersection * cols_intersection) / (rows_union * cols_union)
    return similarity


def _L0_norm(mtx: np.ndarray, indicate_mtx) -> int:
    inv_mtx = np.logical_not(mtx)
    inv_indicate_mtx = np.logical_not(indicate_mtx)
    print(np.sum(inv_mtx & inv_indicate_mtx))
    print(np.sum(mtx & indicate_mtx))
    return (np.sum(inv_mtx & inv_indicate_mtx) + np.sum(mtx & indicate_mtx)) / (mtx.shape[0] * mtx.shape[1])

def extract_mat(adata):
    mat = adata.X.copy() if hasattr(adata.X, 'copy') else np.array(adata.X)
    if sparse.issparse(mat):
        mat = mat.toarray()
    elif isinstance(mat, np.matrix):
        mat = np.array(mat)
    return mat

def nonsparseness_score(adata, bicluster) -> float:
    """
    Calculate nonsparseness score of a bicluster.
    """
    mtx = extract_mat(adata[bicluster.cell_index, :][:, bicluster.gene_index])
    indicate = mtx > 0
    return np.sum(indicate) / (mtx.shape[0] * mtx.shape[1])

def cliff_score(adata, bicluster) -> float:
    """
    Calculate Cliff's delta score of a bicluster.
    """
    return np.mean(cliff_delta_bicluster(adata, bicluster))


def relevance_score(adata, bicluster) -> float:
    gene_mtx = extract_mat(adata[:, bicluster.gene_index])
    gene_var = np.var(gene_mtx, axis=0) 
    in_mtx = extract_mat(adata[bicluster.cell_index, :][:, bicluster.gene_index])
    var_in_mtx = np.var(in_mtx, axis=0) 
    score = 1 - np.mean(var_in_mtx / (gene_var + 1e-12))
    return score

def calculate_acv(adata, bicluster) -> float:
    data_matrix = extract_mat(adata)
    rows = bicluster.cell_index
    cols = bicluster.gene_index
    num_rows, num_cols = len(rows), len(cols)
    if len(rows) <= 1 or len(cols) <= 1:
        return 0
    row_corr_sum = 0
    for i1 in rows:
        for i2 in rows:
            if i1 != i2:
                corr = np.corrcoef(data_matrix[i1, :], data_matrix[i2, :])[0, 1]
                if np.isnan(corr):
                    corr = 0
                row_corr_sum += abs(corr)
    
    col_corr_sum = 0
    for j1 in cols:
        for j2 in cols:
            if j1 != j2: 
                corr = np.corrcoef(data_matrix[:, j1], data_matrix[:, j2])[0, 1]
                if np.isnan(corr):
                    corr = 0
                col_corr_sum += abs(corr)
    
    row_acv = (row_corr_sum - num_rows) / (num_rows**2 - num_rows) if num_rows > 1 else 0
    
    col_acv = (col_corr_sum - num_cols) / (num_cols**2 - num_cols) if num_cols > 1 else 0

    return max(row_acv, col_acv)

def eval_bicluster(adata, biclusters, identifier, output_base, **eval_args):
    results = []
    for i, bicluster in enumerate(biclusters):
        result = {}
        result["bicluster"] = i
        result["cells"] = len(bicluster.cell_index)
        result["genes"] = len(bicluster.gene_index)
        print(f"Bicluster {i}: {len(bicluster.cell_index)} cells, {len(bicluster.gene_index)} genes")
        # cliff = algorithms.cliff_delta_bicluster(adata, bicluster)
        # print("Cliff:")
        # print(cliff)
        # print("Max", max(cliff), "Min", min(cliff), "Mean", np.mean(cliff))
        # result["cliff_mean"] = np.mean(cliff)
        mann_stat, mann_p = one_side_mannwhitneyu_bicluster(adata, bicluster)
        print("Mann:")
        print("Max", max(mann_p), "Min", min(mann_p), "Mean", np.mean(mann_p))
        result["mann_mean"] = np.mean(mann_p)
        comb_res = stats.combine_pvalues(mann_p, method='stouffer', weights=np.ones(len(mann_p)))
        combined_p = comb_res.pvalue
        print("Combined p-value:", combined_p)
        result["mann_combined"] = combined_p
        nonsparse = nonsparseness_score(adata, bicluster)
        print("Nonsparseness", nonsparse)
        ri = relevance_score(adata, bicluster)
        print("Relevance", ri)
        acv = calculate_acv(adata, bicluster)
        print("ACV", acv)
        result["nonsparseness"] = nonsparse
        result["relevance"] = ri
        result["acv"] = acv
        results.append(result)
    results_df = pd.DataFrame(results)
    print(compute_coverage(adata, biclusters))
    plot_biclusters_umap(adata, biclusters, save_path=output_base + f"bicluster_umap_{identifier}.png")
    plot_biclusters_heatmap(adata, biclusters, save_path=output_base + f"bicluster_heatmap_{identifier}.png")
    results_df.to_csv(output_base + f"results_{identifier}.csv")
    return results_df