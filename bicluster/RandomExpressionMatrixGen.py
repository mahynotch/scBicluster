import numpy as np
import random as rdm
from typing import Tuple, List, Dict, Optional, Union

def rand_bicluster_expression_matrix_gen(
    matrix_size: Tuple[int, int], 
    num_submatrices: int, 
    num_markers: int = 5, 
    is_count: bool = True, 
    overlap_row: bool = False, 
    overlap_col: bool = False, 
    dropout_rate: float = 0, 
    noise: float = 0, 
    expression_distribution: str = "negative_binomial",
    marker_intensity: float = 2.0,
    background_intensity: float = 0.5,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Generate a random bicluster expression matrix with submatrices that have marker genes.
    
    Parameters:
    ----------
    matrix_size : Tuple[int, int]
        Size of the matrix (rows, columns) representing (cells, genes).
    num_submatrices : int
        Number of submatrices to generate, representing distinct cell populations.
    num_markers : int, optional
        Number of marker genes per submatrix.
    is_count : bool, optional
        If True, generate integer count data. If False, generate normalized expression values.
    overlap_row : bool, optional
        Whether submatrices can overlap in the row dimension (cells).
    overlap_col : bool, optional
        Whether submatrices can overlap in the column dimension (genes).
    dropout_rate : float, optional
        Fraction of values to be set to zero, simulating dropout events in single-cell data.
    noise : float, optional
        Scale of noise to add to the matrix.
    expression_distribution : str, optional
        Distribution to use for expression values ('negative_binomial', 'lognormal', 'normal').
    marker_intensity : float, optional
        Intensity multiplier for marker genes relative to non-markers.
    background_intensity : float, optional
        Intensity for background expression.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns:
    -------
    Tuple[np.ndarray, Dict]
        The generated expression matrix and a dictionary with information about submatrices.
    """
    # Validate inputs
    if not (0 <= dropout_rate <= 1):
        raise ValueError("dropout_rate must be between 0 and 1")
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if num_markers < 0:
        raise ValueError("num_markers must be non-negative")
    if num_submatrices < 1:
        raise ValueError("num_submatrices must be at least 1")
    
    # Set random seeds if provided
    if seed is not None:
        np.random.seed(seed)
        rdm.seed(seed)
    
    num_cells, num_genes = matrix_size
    
    # Create base matrix with background noise
    matrix = np.random.uniform(0, background_intensity, matrix_size)
    
    # Parameters for submatrix sizes - for cell clusters
    min_height = max(5, num_cells // (5 * num_submatrices))
    max_height = max(min_height, num_cells // 3)
    min_width = max(5, num_genes // (5 * num_submatrices))
    max_width = max(min_width, num_genes // 3)
    
    # Track submatrix locations
    submatrix_info = []
    
    # Strategy for placing submatrices (cell clusters)
    attempts = 0
    max_attempts = num_submatrices * 100
    placed_submatrices = 0
    
    while placed_submatrices < num_submatrices and attempts < max_attempts:
        attempts += 1
        
        # Randomly determine submatrix dimensions
        height = np.random.randint(min_height, max_height + 1)  # Cell cluster size
        width = np.random.randint(min_width, max_width + 1)     # Number of genes involved
        
        # Place submatrix using an improved strategy
        if placed_submatrices == 0 or (overlap_row and overlap_col):
            # Random placement for first or fully overlapping submatrices
            row_start = np.random.randint(0, num_cells - height + 1)  # Cell start
            col_start = np.random.randint(0, num_genes - width + 1)   # Gene start
        else:
            # Try to place non-overlapping matrices
            if not overlap_row and not overlap_col:
                # Use multiple attempts to find non-overlapping position
                found_position = False
                for _ in range(50):  # Try 50 times to find a position
                    row_start = np.random.randint(0, num_cells - height + 1)
                    col_start = np.random.randint(0, num_genes - width + 1)
                    
                    # Check if this position overlaps with existing submatrices
                    overlaps = False
                    for info in submatrix_info:
                        prev_row_start, prev_row_end, prev_col_start, prev_col_end = info['bounds']
                        row_end = row_start + height
                        col_end = col_start + width
                        
                        # Check for any overlap
                        if (prev_row_start < row_end and prev_row_end > row_start and 
                            prev_col_start < col_end and prev_col_end > col_start):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        found_position = True
                        break
                
                if not found_position:
                    continue  # Skip this attempt
            else:
                # Semi-random placement for partial overlap cases
                row_start = np.random.randint(0, num_cells - height + 1)
                col_start = np.random.randint(0, num_genes - width + 1)
        
        row_end = row_start + height
        col_end = col_start + width
        
        # Validate position (check for disallowed overlaps)
        valid_position = True
        for info in submatrix_info:
            prev_row_start, prev_row_end, prev_col_start, prev_col_end = info['bounds']
            
            # Check for row overlap (cells)
            row_overlap = max(0, min(row_end, prev_row_end) - max(row_start, prev_row_start)) > 0
            if not overlap_row and row_overlap:
                valid_position = False
                break
                
            # Check for column overlap (genes)
            col_overlap = max(0, min(col_end, prev_col_end) - max(col_start, prev_col_start)) > 0
            if not overlap_col and col_overlap:
                valid_position = False
                break
        
        if valid_position:
            # Choose marker genes for this submatrix - markers are columns (genes)
            marker_cols = rdm.sample(range(col_start, col_end), min(num_markers, width))
            
            # Record submatrix information
            submatrix_info.append({
                'bounds': (row_start, row_end, col_start, col_end),
                'markers': marker_cols
            })
            
            # Generate values for the submatrix
            generate_submatrix(
                matrix, 
                row_start, row_end, 
                col_start, col_end, 
                marker_cols,
                distribution=expression_distribution,
                marker_intensity=marker_intensity
            )
            
            placed_submatrices += 1
    
    if placed_submatrices < num_submatrices:
        print(f"Warning: Only placed {placed_submatrices} out of {num_submatrices} requested submatrices.")
    
    # Add global noise if specified
    if noise > 0:
        if expression_distribution == 'normal':
            noise_matrix = np.random.normal(0, noise, matrix_size)
        else:
            noise_matrix = np.random.exponential(noise, matrix_size)
            neg_mask = np.random.random(matrix_size) < 0.5
            noise_matrix[neg_mask] *= -1
        
        matrix += noise_matrix
        matrix = np.maximum(0, matrix)  # Ensure non-negative values
    
    # Apply dropout if specified (important for single-cell data)
    if dropout_rate > 0:
        dropout_mask = np.random.random(matrix_size) < dropout_rate
        matrix[dropout_mask] = 0
    
    # Convert to counts or normalize
    if is_count:
        matrix = np.round(matrix).astype(int)
    else:
        normalize_matrix(matrix)
    
    return matrix, {'submatrices': submatrix_info}

def generate_submatrix(
    matrix: np.ndarray, 
    row_start: int, 
    row_end: int, 
    col_start: int, 
    col_end: int, 
    marker_cols: List[int],
    distribution: str = 'negative_binomial',
    marker_intensity: float = 2.0
):
    """
    Generate values for a submatrix with the specified distribution and marker genes.
    
    In this context:
    - Rows represent cells
    - Columns represent genes
    - marker_cols are the marker genes (columns) for this cell cluster
    """
    height = row_end - row_start  # Number of cells
    
    for j in range(col_start, col_end):
        is_marker = j in marker_cols
        
        # Generate different expression patterns based on gene type
        if distribution == 'negative_binomial':
            # Parameters for negative binomial
            if is_marker:
                r, p = 2.0, 0.2  # Higher expression for marker genes
                intensity = marker_intensity
            else:
                r, p = 1.0, 0.1  # Lower expression for non-marker genes
                intensity = 1.0
            
            values = np.random.negative_binomial(r, p, size=height) * intensity
            matrix[row_start:row_end, j] = values
                
        elif distribution == 'lognormal':
            # Parameters for lognormal
            if is_marker:
                mean, sigma = 1.5, 0.7  # Higher expression for marker genes
            else:
                mean, sigma = 0.5, 0.5  # Lower expression for non-marker genes
            
            values = np.random.lognormal(mean, sigma, size=height)
            if not is_marker:
                values = values / marker_intensity  # Reduce non-marker expression
            matrix[row_start:row_end, j] = values
            
        else:  # Default to normal distribution
            if is_marker:
                mean, std = 5.0, 1.0  # Higher expression for marker genes
            else:
                mean, std = 2.0, 0.5  # Lower expression for non-marker genes
            
            values = np.maximum(0, np.random.normal(mean, std, size=height))
            matrix[row_start:row_end, j] = values

def normalize_matrix(matrix: np.ndarray, target_sum: float = 1e6):
    """
    Normalize the matrix to simulate library size normalization.
    For single-cell data where rows=cells, columns=genes.
    """
    # Simulate library size variation between cells
    row_size_factors = np.random.lognormal(0, 0.5, size=matrix.shape[0])
    row_size_factors = row_size_factors / np.median(row_size_factors)
    
    # Apply size factors to cells (rows)
    for i in range(matrix.shape[0]):
        matrix[i, :] *= row_size_factors[i]
    
    # Library size normalization (similar to CPM)
    row_sums = np.sum(matrix, axis=1)
    row_sums_nonzero = np.where(row_sums > 0, row_sums, 1)
    
    for i in range(matrix.shape[0]):
        if row_sums[i] > 0:
            matrix[i, :] = matrix[i, :] * target_sum / row_sums_nonzero[i]