from .utils import CustomBiclusterClass, BiclusterList
import scanpy as sc
from scipy import stats
from scanpy import AnnData
import numpy as np

def generate_random_biclusters(adata, min_row=10, min_col=10, num_clusters=5, rnd_state=0):
    """
    Generate random biclusters from existing AnnData object.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object containing the data matrix.
    min_row : int, default=10
        Minimum number of rows (cells/observations) in each bicluster.
    min_col : int, default=10
        Minimum number of columns (genes/variables) in each bicluster.
    num_clusters : int, default=5
        Number of biclusters to generate.
    rnd_state : int, default=0
        Random state for reproducibility.
        
    Returns:
    --------
    biclusters : list of tuples
        List of (rows, cols) index arrays for each bicluster.
    """
    np.random.seed(rnd_state)
    
    n_rows, n_cols = adata.shape
    
    if min_row > n_rows:
        raise ValueError(f"min_row ({min_row}) exceeds number of rows in data ({n_rows})")
    if min_col > n_cols:
        raise ValueError(f"min_col ({min_col}) exceeds number of columns in data ({n_cols})")
    
    biclusters = []
    for i in range(num_clusters):
        bicluster_n_rows = np.random.randint(min_row, max(min_row+1, n_rows//2))
        bicluster_n_cols = np.random.randint(min_col, max(min_col+1, n_cols//2))
        
        row_indices = np.random.choice(n_rows, size=bicluster_n_rows, replace=False)
        col_indices = np.random.choice(n_cols, size=bicluster_n_cols, replace=False)
        
        biclusters.append((row_indices, col_indices))
    
    return biclusters

def infer_biclusters(adata, bicluster: CustomBiclusterClass, run_func, run_args) -> BiclusterList:
    """
    Iteratively infer into each bicluster using the provided model.
    """
    biclusters = BiclusterList()
    adata_sub = adata[bicluster.cell_index, bicluster.gene_index]
    model = run_func(adata_sub, **run_args)
    for bicustering in model:
        biclusters.append(bicluster.cell_index[bicustering.cell_index], bicluster.gene_index[bicustering.gene_index])
    return biclusters

def auto_infer_biclusters(adata, biclusters: BiclusterList, thr, eval_func, eval_args, infer_func, infer_args) -> BiclusterList:
    """
    Iteratively infer into each bicluster using the provided model.
    """
    new_biclusters = BiclusterList()
    for bicluster in biclusters:
        if eval_func(adata, bicluster, **eval_args):
            new_biclusters.append(bicluster)
        else:
            new_biclusters.extend(infer_biclusters(adata, bicluster, infer_func, infer_args))
    return new_biclusters

def return_genes_from_bicluster(adata: AnnData, bicluster: CustomBiclusterClass):
    """
    Return the gene names of a bicluster.
    """
    var_names = adata.var_names
    return var_names[bicluster.gene_index], var_names[np.setdiff1d(np.arange(adata.n_vars), bicluster.gene_index)]

def return_cells_from_bicluster(adata: AnnData, bicluster: CustomBiclusterClass):
    """
    Return the cell names of a bicluster.
    """
    obs_names = adata.obs_names
    return obs_names[bicluster.cell_index], obs_names[np.setdiff1d(np.arange(adata.n_obs), bicluster.cell_index)]

def t_test_bicluster(adata: AnnData, bicluster: CustomBiclusterClass):
    """
    Perform t-test on each gene in the bicluster, return a list of p-values for each gene.
    """
    if np.array_equal(np.arange(adata.n_obs), bicluster.cell_index) or len(bicluster.gene_index) == 0 or len(bicluster.cell_index) == 0:
        return np.zeros(adata.n_vars).flatten().tolist()
    expr_data = adata.X.toarray()
    return np.array([sc.tl.rank_gene_groups(expr_data[:, i], groups=bicluster.cell_index, method='t-test', use_raw=False)[0] for i in bicluster.gene_index])


def cliffs_delta(x, y):
    greater = 0
    less = 0
    for i in range(len(x)):
        for j in range(len(y)):
            if x[i] > y[j]:
                greater += 1
            elif x[i] < y[j]:
                less += 1
    
    return (greater - less) / (len(x) * len(y))

def cliff_delta_bicluster(adata: AnnData, bicluster: CustomBiclusterClass):
    if np.array_equal(np.arange(adata.n_obs), bicluster.cell_index) or len(bicluster.gene_index) == 0 or len(bicluster.cell_index) == 0:
        return np.zeros(adata.n_vars).flatten().tolist()
    expr_data = adata.X.toarray() 
    x, y = expr_data[bicluster.cell_index, :], expr_data[bicluster.get_negative_cell_index(np.arange(adata.n_obs)), :]
    return np.array([cliffs_delta(x[:, i], y[:, i]) for i in bicluster.gene_index])

def wilcoxon_bicluster(adata: AnnData, bicluster: CustomBiclusterClass):
    """
    Perform Wilcoxon rank-sum test on each gene in the bicluster, return a list of p-values for each gene.
    """
    if np.array_equal(np.arange(adata.n_obs), bicluster.cell_index) or len(bicluster.gene_index) == 0 or len(bicluster.cell_index) == 0:
        return np.zeros(adata.n_vars).flatten().tolist()
    expr_data = adata.X.toarray()
    return np.array([sc.tl.rank_gene_groups(expr_data[:, i], groups=bicluster.cell_index, method='wilcoxon', use_raw=False)[0] for i in bicluster.gene_index])

def one_side_mannwhitneyu_bicluster(adata: AnnData, bicluster: CustomBiclusterClass):
    """
    Perform one-sided Mann-Whitney U test on each gene in the bicluster, return a list of p-values for each gene.
    """
    if np.array_equal(np.arange(adata.n_obs), bicluster.cell_index) or len(bicluster.gene_index) == 0 or len(bicluster.cell_index) == 0:
        return np.zeros(adata.n_vars).flatten().tolist()
    expr_data = adata.X.toarray()
    x, y = expr_data[bicluster.cell_index, :], expr_data[bicluster.get_negative_cell_index(np.arange(adata.n_obs)), :]
    res = np.array([stats.mannwhitneyu(x[:, i], y[:, i], alternative="greater") for i in bicluster.gene_index])
    return res[:, 0].flatten(), res[:, 1].flatten()

def count_all_fold_change(adata: AnnData, bicluster: CustomBiclusterClass):
    """
    Summarize the fold change of all genes in each bicluster, return a list of fold changes for each bicluster.
    """
    if np.array_equal(np.arange(adata.n_obs), bicluster.cell_index) or len(bicluster.gene_index) == 0 or len(bicluster.cell_index) == 0:
        return np.zeros(adata.n_vars).flatten().tolist()
    expr_data = adata.X.toarray()
    return np.log(np.array(np.mean(expr_data[bicluster.cell_index, :], axis=0) + 1e-8) / (np.mean(expr_data[np.setdiff1d(np.arange(adata.n_obs), bicluster.cell_index), :], axis=0) + 1e-8).flatten().tolist())

def count_all_fold_change(adata: AnnData, bicluster):
    """
    Summarize the fold change of all genes in each bicluster, return a list of fold changes for each bicluster.
    """
    if np.array_equal(np.arange(adata.n_obs), bicluster.cell_index) or len(bicluster.gene_index) == 0 or len(bicluster.cell_index) == 0:
        return np.zeros(adata.n_vars).flatten().tolist()
    expr_data = adata.X.toarray()
    return np.log(np.array(np.mean(expr_data[bicluster.cell_index, :], axis=0) + 1e-8) / (np.mean(expr_data[np.setdiff1d(np.arange(adata.n_obs), bicluster.cell_index), :], axis=0) + 1e-8).flatten().tolist())

def count_incluster_fold_change(adata: AnnData, bicluster):
    """
    Summarize the fold change of genes in each bicluster, return a list of fold changes for each bicluster.
    """
    if np.array_equal(np.arange(adata.n_obs), bicluster.cell_index) or len(bicluster.gene_index) == 0 or len(bicluster.cell_index) == 0:
        return np.zeros(adata.n_vars).flatten().tolist()
    expr_data = adata.X.toarray()
    return np.log(np.array((np.mean(expr_data[bicluster.cell_index, :][:, bicluster.gene_index], axis=0) + 1e-8) / (np.mean(expr_data[np.setdiff1d(np.arange(adata.n_obs), bicluster.cell_index), :][:, bicluster.gene_index], axis=0) + 1e-8)).flatten().tolist())

def compute_coverage(adata: AnnData, biclusters: BiclusterList):
    """
    Compute the coverage of each bicluster, return a list of coverage for each bicluster.
    """
    base_bicluster = CustomBiclusterClass(cell_index=np.array([]), gene_index=np.array([]))
    for bicluster in biclusters:
        base_bicluster = base_bicluster.union(bicluster)
    cell_cover = len(base_bicluster.cell_index) / adata.n_obs
    gene_cover = len(base_bicluster.gene_index) / adata.n_vars
    return cell_cover, gene_cover

def extract_most_relavent_positive_negative_genes(adata: AnnData, bicluster: CustomBiclusterClass, n_genes=10):
    """
    Extract the most relevant positive and negative genes in the bicluster.
    """
    if np.array_equal(np.arange(adata.n_obs), bicluster.cell_index) or len(bicluster.gene_index) == 0 or len(bicluster.cell_index) == 0:
        return np.zeros(adata.n_vars).flatten().tolist()
    expr_data = adata.X.toarray()
    score = count_incluster_fold_change(adata, bicluster)
    positive_genes = bicluster.gene_index[np.argsort(score)[-n_genes:]]
    negative_genes = bicluster.gene_index[np.argsort(score)[:n_genes]]
    return adata.var["feature_name"][positive_genes], adata.var["feature_name"][negative_genes]