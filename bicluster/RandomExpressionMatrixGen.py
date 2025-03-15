import random

import numpy as np
import numpy.random as npr
from scipy.sparse import lil_matrix
import random as rdm
import matplotlib.pyplot as plt


def rand_bicluster_expression_matrix_gen(matrix_size: tuple[int, int], num_submatrices: int, num_markers: int=5, overlap_row: bool=False, overlap_col: bool=False, dropout_rate=0, backgroud_noise=0, seed: int=42) -> np.ndarray:
    """
    Generate a random bicluster expression matrix with submatrices that have marker genes.
    
    Parameters:
    ----------
    matrix_size : tuple[int, int]
        Size of the matrix (rows, columns).
    num_submatrices : int
        Number of submatrices to generate.
    num_markers : int, optional
        Number of marker genes per submatrix.
    overlap_row : bool, optional
        Whether submatrices can overlap in the row dimension.
    overlap_col : bool, optional
        Whether submatrices can overlap in the column dimension.
    dropout_rate : float, optional
        Fraction of values to be set to zero.
    backgroud_noise : float, optional
        Scale of Gaussian noise to add to the matrix.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns:
    -------
    np.ndarray
        The generated expression matrix.
    """
    if seed != None:
        npr.seed(seed)
        rdm.seed(seed)
    
    size_x = matrix_size[0]
    size_y = matrix_size[1]
    
    matrix = np.zeros(matrix_size, dtype=float)
    

    submatrix_min_size_x = max(10, size_x // num_submatrices)
    submatrix_max_size_x = max(10, size_x // 2)
    submatrix_min_size_y = max(10, size_y // num_submatrices)
    submatrix_max_size_y = max(10, size_y // 2)
    
    occupied_areas = []
    min_gap = 2
    attempts = 0
    max_attempts = num_submatrices * 100  
    submatrices_placed = 0
    
    old_row_end = 0
    old_col_end = 0
    
    while submatrices_placed < num_submatrices:
        attempts += 1
        if attempts > max_attempts:
            print("Failed to place all submatrices, max attempts reached.")
            break
        submatrix_size_x = npr.randint(submatrix_min_size_x, submatrix_max_size_x + 1)
        submatrix_size_y = npr.randint(submatrix_min_size_y, submatrix_max_size_y + 1)
        
        if overlap_row and overlap_col:
            row_start = npr.randint(0, size_x - submatrix_size_x + 1)
            col_start = npr.randint(0, size_y - submatrix_size_y + 1)
        elif overlap_row:
            row_start = npr.randint(0, size_x - submatrix_size_x + 1)
            col_start = old_col_end
        elif overlap_col:
            row_start = old_row_end
            col_start = npr.randint(0, size_y - submatrix_size_y + 1)
        else:
            row_start = old_row_end
            col_start = old_col_end
        
        row_end = row_start + submatrix_size_x
        col_end = col_start + submatrix_size_y
        
        valid_position = True
        if not overlap_row or not overlap_col:
            for area in occupied_areas:
                area_row_start, area_row_end, area_col_start, area_col_end = area
                
                row_overlap = not (row_end < area_row_start - min_gap or row_start > area_row_end + min_gap)
                if not overlap_row and row_overlap:
                    valid_position = False
                    break
                
                col_overlap = not (col_end < area_col_start - min_gap or col_start > area_col_end + min_gap)
                if not overlap_col and col_overlap:
                    valid_position = False
                    break
        
        if valid_position:
            occupied_areas.append((row_start, row_end, col_start, col_end))
            
            generate_submatrix(matrix, row_start, col_start, submatrix_size_x, submatrix_size_y, num_markers)
            
            submatrices_placed += 1
            old_row_end = row_end
            old_col_end = col_end
    
    if backgroud_noise > 0:
        noise = npr.normal(0, backgroud_noise, matrix_size)
        matrix += noise
        matrix[matrix < 0] = 0
    
    if dropout_rate > 0:
        mask = npr.random(matrix_size) < dropout_rate
        matrix[mask & (matrix > 0)] = 0
    normalize_rows(matrix)
    
    return matrix

def generate_submatrix(matrix, row_start, col_start, size_x, size_y, num_markers):
    """Generate a submatrix with marker genes."""
    marker_cols = npr.choice(range(size_y), size=min(num_markers, size_y), replace=False)
    
    for j in range(size_y):
        col_idx = col_start + j
    
        if j in marker_cols:
            mean = 5.0
            std = 1.0
        else:
            mean = 1.0
            std = 0.5
        
        values = npr.normal(loc=mean, scale=std, size=size_x)
        values = np.maximum(0, values)
        matrix[row_start:row_start+size_x, col_idx] = values

def normalize_rows(matrix):
    """Normalize rows to have approximately the same sum."""
    row_sums = np.sum(matrix, axis=1)
    non_zero_mask = row_sums > 0
    if np.any(non_zero_mask):
        target_sum = np.mean(row_sums[non_zero_mask])
        for i in range(matrix.shape[0]):
            if row_sums[i] > 0:
                matrix[i, :] *= target_sum / row_sums[i]