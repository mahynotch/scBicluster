import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import scanpy as sc
import pickle
import time
from datetime import datetime
import threading
import functools
# from kneed import KneeLocator

class CustomBiclusterClass:
    def __init__(self, bicluster=None, cell_index=None, gene_index=None):
        if bicluster is not None:
            self.cell_index: np.ndarray = bicluster.rows
            self.gene_index: np.ndarray = bicluster.cols
        else:
            self.cell_index: np.ndarray = cell_index
            self.gene_index: np.ndarray = gene_index
    
    def row_len(self):
        return len(self.cell_index)
    
    def col_len(self):
        return len(self.gene_index)
    
    def get_negative_gene_index(self, col_index):
        return np.setdiff1d(col_index, self.gene_index)
    
    def get_negative_cell_index(self, row_index):
        return np.setdiff1d(row_index, self.cell_index)
    
    def union(self, other):
        return CustomBiclusterClass(cell_index=np.union1d(self.cell_index, other.cell_index), gene_index=np.union1d(self.gene_index, other.gene_index))
    
    def intersection(self, other):
        return CustomBiclusterClass(cell_index=np.intersect1d(self.cell_index, other.cell_index), gene_index=np.intersect1d(self.gene_index, other.gene_index))
    
    def area(self):
        return self.row_len() * self.col_len()
    
    def sort(self):
        self.cell_index.sort()
        self.gene_index.sort()
        
    def substraction(self, other):
        return CustomBiclusterClass(cell_index=np.setdiff1d(self.cell_index, other.cell_index), gene_index=np.setdiff1d(self.gene_index, other.gene_index))
    
    def overlap(self, other):
        min_area = min(self.row_len() * self.col_len(), other.row_len() * other.col_len())
        return self.intersection(other).row_len() * self.intersection(other).col_len() / min_area
    
    def __eq__(self, other):
        return self.cell_index == other.cell_index and self.gene_index == other.gene_index
    
    def __add__(self, other):
        return self.union(other)
    
    def __sub__(self, other):
        return self.substraction(other)
    
    def __lt__(self, other):
        return self.area() < other.area()

    def __le__(self, other):
        return self.area() <= other.area()

    def __gt__(self, other):
        return self.area() > other.area()

    def __ge__(self, other):
        return self.area() >= other.area()
    
    def __str__(self):
        return " nRows: " + str(len(self.cell_index)) + " nCols: " + str(len(self.gene_index))
    
    def __repr__(self):
        return self.__str__()
    
class BiclusterList:
    def __init__(self, biclusters: list[CustomBiclusterClass]=None):
        if biclusters is None:
            self.biclusters = []
        else:
            self.biclusters = biclusters
        
    
    def append(self, row_index, col_index):
        self.biclusters.append(CustomBiclusterClass(cell_index=row_index, gene_index=col_index))
        
    def extend(self, biclusters):
        if type(biclusters) == BiclusterList:
            self.biclusters.extend(biclusters.biclusters)
        elif type(biclusters) == list:
            self.biclusters.extend(biclusters)
            
    def merge(self, thr):
        """
        Test each bicluster with jaccard metrice compared to a threshold. If the bicluster passes the threshold, merge it with the previous bicluster.
        """
        from .metrices import calculate_jaccard
        biclusters = sorted(self.biclusters, key=lambda x: x.cell_index[0])
        new_biclusters = []
        for i, bicluster in enumerate(biclusters):
            if i == 0:
                new_biclusters.append(bicluster)
                continue
            
            prev_bicluster = new_biclusters[-1]
            _, _, similarity = calculate_jaccard(prev_bicluster, bicluster)
            if similarity >= thr:
                new_biclusters[-1] = CustomBiclusterClass(
                    cell_index = np.union1d(prev_bicluster.cell_index, bicluster.cell_index),
                    gene_index = np.union1d(prev_bicluster.gene_index, bicluster.gene_index)
                )
            else:
                new_biclusters.append(bicluster)
        self.biclusters = new_biclusters
        return new_biclusters
        
    def sort(self):
        self.biclusters = sorted(self.biclusters, key=lambda x: x.area(), reverse=True)
        return self.biclusters
    
    def export_biclusters(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.biclusters, f)

    def import_biclusters(self, path):
        with open(path, 'rb') as f:
            self.biclusters = pickle.load(f)

    def combine(self):
        new_bicluster = CustomBiclusterClass()
        for bicluster in self.biclusters:
            new_bicluster = new_bicluster + bicluster
        return new_bicluster
    
    def save_to_adata(self, adata: sc.AnnData, key: str="bicluster"):
        cell_biclusters = [[] for _ in range(adata.n_obs)]
        gene_biclusters = [[] for _ in range(adata.n_vars)]

        for i, bic in enumerate(self.biclusters):
            for c in bic.cell_index:
                cell_biclusters[c].append(i)
            for g in bic.gene_index:
                gene_biclusters[g].append(i)

        adata.obs[key] = cell_biclusters
        adata.var[key] = gene_biclusters
            
    def __setitem__(self, index, value: CustomBiclusterClass):
        self.biclusters[index] = value
        
    def __getitem__(self, index):
        return self.biclusters[index]
    
    def __len__(self):
        return len(self.biclusters)
    
    def __repr__(self):
        return str(self.biclusters)
    
    def __iter__(self):
        return iter(self.biclusters)
    
def sort_biclusters(biclusters):
    return sorted(biclusters, key=lambda x: x.area(), reverse=True)
    
def generate_BiclusterList_from_biclusters(biclusters):
    return BiclusterList([CustomBiclusterClass(bicluster=bic) for bic in biclusters])

def filter_low_expression_row_col(data: np.ndarray, row_threshold: float, col_threshold: float) -> np.ndarray:
    row_sum = np.sum(data, axis=1)
    col_sum = np.sum(data, axis=0)
    row_index = np.where(row_sum >= row_threshold)[0]
    col_index = np.where(col_sum >= col_threshold)[0]
    return data[row_index][:, col_index], row_index, col_index

def find_gap(array: np.ndarray) -> np.ndarray:
    array_length = len(array)
    return array[1:array_length] - array[0:array_length - 1]

def find_elbow_point_kneedle(array: np.array) -> tuple:
    array = np.array(array).flatten()
    if array.ndim == 1:
        sorted_array = np.sort(array)
        gap_gap = find_gap(find_gap(sorted_array))
        elbow_point = np.argmax(gap_gap) + 1
    return elbow_point, sorted_array[elbow_point]

def randomize_matrix(array):
    row_index_array = np.arange(array.shape[0])
    col_index_array = np.arange(array.shape[1])
    np.random.shuffle(row_index_array)
    np.random.shuffle(col_index_array)
    return array[row_index_array][:, col_index_array], row_index_array, col_index_array

def label_disp(label):
    values, counts = np.unique(label, return_counts=True)
    print(f"for {values} the numbers are {counts}, with total of {np.sum(counts)} instances")


def normalize_by_cell(array: np.ndarray):
    min_vals = array.min(axis=0)
    max_vals = array.max(axis=0)
    return (array - min_vals) / (max_vals - min_vals)


def noise_adding(array: np.ndarray, percent_drop: float, seed: int = 0) -> np:
    generator = np.random.Generator(np.random.PCG64(seed))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i, j] = generator.binomial(array[i, j], 1 - percent_drop)
    return array

def _scale_normalize(X, target=1):
    """Normalize ``X`` by scaling rows and columns independently.

    Returns the normalized matrix and the row and column scaling
    factors.
    """
    row_diag = np.asarray(target / np.sqrt(X.sum(axis=1))).squeeze()
    col_diag = np.asarray(target / np.sqrt(X.sum(axis=0))).squeeze()
    row_diag = np.where(np.isnan(row_diag), 0, row_diag)
    col_diag = np.where(np.isnan(col_diag), 0, col_diag)
    an = row_diag[:, np.newaxis] * X * col_diag
    return an, row_diag, col_diag

def bistochastic_normalize(X, max_iter=1000, tol=1e-5, target=1):
    """Normalize ``X`` to be row and column stochastic. Adopted from scikit-learn.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The matrix to be normalized.
    max_iter : int, optional (default: 1000)
        The maximum number of iterations to perform.
    tol : float, optional (default: 1e-5)
        The convergence threshold.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        The normalized matrix.
    """
    X = np.asarray(X)
    X, row_diag, col_diag = _scale_normalize(X, target)
    for i in range(max_iter):
        X, row_diag, col_diag = _scale_normalize(X, target)
        if np.allclose(X.sum(axis=1), target, atol=tol) and np.allclose(X.sum(axis=0), target, atol=tol):
            break
    return X, row_diag, col_diag


class TimeoutError(Exception):
    """Exception raised when a function execution times out."""
    pass

def timed_execution(timeout_seconds=None):
    """
    Decorator that both measures execution time and optionally enforces a timeout.
    
    Parameters:
    -----------
    timeout_seconds : int or None, default=None
        Maximum allowed execution time in seconds.
        If None, the function will run without a timeout.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Print start message
            start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Running {func.__name__} start at {start_datetime}")
            
            if timeout_seconds is None:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                print(f"{func.__name__} completed in {execution_time:.4f} seconds")
                return result
            
            result = [TimeoutError(f"Function '{func.__name__}' timed out after {timeout_seconds} seconds")]
            execution_completed = [False]
            
            def target():
                try:
                    start_time = time.time()
                    result[0] = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    print(f"{func.__name__} completed in {execution_time:.4f} seconds")
                    execution_completed[0] = True
                except Exception as e:
                    result[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            
            start_time = time.time()
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                execution_time = time.time() - start_time
                print(f"{func.__name__} exceeded time limit of {timeout_seconds} seconds (ran for {execution_time:.4f} seconds)")
                raise TimeoutError(f"Function '{func.__name__}' timed out after {timeout_seconds} seconds")
            
            if not execution_completed[0]:
                if isinstance(result[0], Exception):
                    raise result[0]
            
            return result[0]
            
        return wrapper
    return decorator

def extract_mat(adata):
    mat = adata.X.copy() if hasattr(adata.X, 'copy') else np.array(adata.X)
    if scipy.sparse.issparse(mat):
        mat = mat.toarray()
    elif isinstance(mat, np.matrix):
        mat = np.array(mat)
    return mat

def mat_to_adata(mat):
    adata = sc.AnnData(mat)
    return adata

def info_to_biclusters(info):
    biclust_list = BiclusterList()
    submats = info['submatrices']
    marker_list = []
    for submat in submats:
        bound = submat['bounds']
        rows = np.arange(bound[0], bound[1])
        cols = np.arange(bound[2], bound[3])
        biclust_list.append(rows, cols)
        marker = submat['markers']
        if marker is not None:
            marker_list.append(np.array(marker))
    return biclust_list, marker_list

def shuffle_index_apply_biclusters(biclusters, row_index_array, col_index_array):
    new_biclusters = BiclusterList()
    
    # Create forward mappings (where did each original index go?)
    row_forward_map = np.zeros_like(row_index_array, dtype=int)
    for i, val in enumerate(row_index_array):
        row_forward_map[val] = i
    
    col_forward_map = np.zeros_like(col_index_array, dtype=int)
    for i, val in enumerate(col_index_array):
        col_forward_map[val] = i
    
    for bicluster in biclusters:
        # Apply forward mapping to find new positions
        new_cell_indices = row_forward_map[bicluster.cell_index]
        new_gene_indices = col_forward_map[bicluster.gene_index]
        new_biclusters.append(new_cell_indices, new_gene_indices)
    return new_biclusters


def shuffle_index_apply_markers(marker_list, col_index_array):
    new_marker_list = []
    
    # Create forward mapping
    col_forward_map = np.zeros_like(col_index_array, dtype=int)
    for i, val in enumerate(col_index_array):
        col_forward_map[val] = i
    
    for markers in marker_list:
        new_marker = col_forward_map[markers]
        new_marker_list.append(new_marker)
    return new_marker_list



def binarize_expr_array(array: np.ndarray):
    array - np.max(array, axis=0) / 2
    # TODO