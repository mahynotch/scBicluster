import numpy as np
from typing import List
from .algorithms import cliff_delta_bicluster, one_side_mannwhitneyu_bicluster, compute_coverage
from .visualization import plot_biclusters_umap, plot_biclusters_heatmap
import pandas as pd
from scipy import stats, sparse
import matplotlib.pyplot as plt

def calculate_jaccard(biclust1, biclust2):
    # Calculate for rows (cells)
    rows1 = set(biclust1.cell_index)
    rows2 = set(biclust2.cell_index)
    row_jaccard = len(rows1.intersection(rows2)) / max(1, len(rows1.union(rows2)))
    
    # Calculate for columns (genes)
    cols1 = set(biclust1.gene_index)
    cols2 = set(biclust2.gene_index)
    col_jaccard = len(cols1.intersection(cols2)) / max(1, len(cols1.union(cols2)))
    
    # Calculate for the entire bicluster (cell-gene pairs)
    pairs1 = {(r, c) for r in rows1 for c in cols1}
    pairs2 = {(r, c) for r in rows2 for c in cols2}
    bicluster_jaccard = len(pairs1.intersection(pairs2)) / max(1, len(pairs1.union(pairs2)))
    
    return row_jaccard, col_jaccard, bicluster_jaccard


def L0_norm(mtx: np.ndarray, indicate_mtx) -> int:
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

    
def consensus_matrix_analysis(predicted_biclusters, target_biclusters, adata, output_dir):
    """
    Create and compare consensus matrices showing how often cells appear together
    """
    n_cells = adata.n_obs
    
    # Build consensus matrices
    def build_consensus(biclusters):
        matrix = np.zeros((n_cells, n_cells))
        for bc in biclusters:
            for i in bc.cell_index:
                for j in bc.cell_index:
                    matrix[i, j] += 1
        # Normalize
        if len(biclusters) > 0:
            matrix /= len(biclusters)
        return matrix
    
    pred_consensus = build_consensus(predicted_biclusters)
    target_consensus = build_consensus(target_biclusters)
    
    # Calculate similarity
    flat_pred = pred_consensus.flatten()
    flat_target = target_consensus.flatten()
    correlation = np.corrcoef(flat_pred, flat_target)[0, 1]
    mse = np.mean((flat_pred - flat_target)**2)
    
    vmin = 0
    vmax = np.quantile(flat_target, 0.99)
    
    # Visualize
    plt.figure(figsize=(12, 6))
    
    # Visualize
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(target_consensus, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.title("Target Consensus Matrix", fontdict={"fontsize": 18})
    plt.colorbar(im1)  # Pass the image object to colorbar

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(pred_consensus, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.title(f"Predicted Consensus Matrix\nCorr={correlation:.2f}, MSE={mse:.4f}", fontdict={"fontsize": 18})
    plt.colorbar(im2)  # Pass the image object to colorbar
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/consensus_comparison.png")
    plt.close()
    
    return {"correlation": correlation, "mse": mse}

# 2. Recovery and Relevance
def calculate_recovery_relevance(pred, target):
    pred_elements = {(r, c) for r in pred.cell_index for c in pred.gene_index}
    target_elements = {(r, c) for r in target.cell_index for c in target.gene_index}
    
    intersection = len(pred_elements.intersection(target_elements))
    recovery = intersection / max(1, len(target_elements))  # How much of target was found
    relevance = intersection / max(1, len(pred_elements))   # How precise the prediction is
    
    f1_score = 2 * (recovery * relevance) / max(1e-10, (recovery + relevance))
    return recovery, relevance, f1_score
    
def evaluate_biclusters_to_target(predicted_biclusters, target_biclusters):
    """Evaluate predicted biclusters against target biclusters"""
    results = {}
    
    
    # Find best matching pairs between predicted and target biclusters
    best_matches = []
    for i, pred in enumerate(predicted_biclusters):
        best_match = -1
        best_score = -1
        
        for j, target in enumerate(target_biclusters):
            _, _, bicluster_jaccard = calculate_jaccard(pred, target)
            if bicluster_jaccard > best_score:
                best_score = bicluster_jaccard
                best_match = j
        
        best_matches.append((i, best_match, best_score))
    
    # Calculate detailed metrics for each match
    match_metrics = []
    for pred_idx, target_idx, score in best_matches:
        if score > 0:  # Only evaluate non-zero matches
            pred = predicted_biclusters[pred_idx]
            target = target_biclusters[target_idx]
            
            row_j, col_j, bic_j = calculate_jaccard(pred, target)
            rec, rel, f1 = calculate_recovery_relevance(pred, target)
            
            match_metrics.append({
                'pred_idx': pred_idx, 
                'target_idx': target_idx,
                'jaccard': bic_j,
                'row_jaccard': row_j,
                'col_jaccard': col_j, 
                'recovery': rec,
                'relevance': rel,
                'f1_score': f1
            })
    
    # Calculate overall metrics
    results['matches'] = match_metrics
    results['overall'] = {
        'avg_jaccard': np.mean([m['jaccard'] for m in match_metrics]) if match_metrics else 0,
        'avg_f1': np.mean([m['f1_score'] for m in match_metrics]) if match_metrics else 0,
        'avg_recovery': np.mean([m['recovery'] for m in match_metrics]) if match_metrics else 0,
        'avg_relevance': np.mean([m['relevance'] for m in match_metrics]) if match_metrics else 0,
        'match_rate': len(match_metrics) / len(predicted_biclusters) if predicted_biclusters else 0
    }
    
    return results