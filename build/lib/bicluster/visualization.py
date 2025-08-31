from .utils import CustomBiclusterClass, BiclusterList
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scipy
import pandas as pd
import plotly.graph_objects as go
import random
import colorsys
import ast
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

def plot_biclusters_umap(adata: sc.AnnData, biclusters: list[CustomBiclusterClass], save_path=None):
    """
    Plot UMAP plot grouped by each bicluster. 
    """
    if len(biclusters) == 0:
        raise ValueError("No bicluster is provided.")
    # If no UMAP is computed, compute it
    if "X_umap" not in adata.obsm.keys():
        sc.pp.neighbors(adata, use_rep="X")
        sc.tl.umap(adata)
    
    # Plot UMAP plot for each bicluster
    vertical = int(np.sqrt(len(biclusters)))
    horizontal = int(np.ceil(len(biclusters) / vertical))
    fig, axs = plt.subplots(vertical, horizontal, figsize=(30, 30))
    for i, bicluster in enumerate(biclusters):
        if axs.ndim == 2:
            ax = axs[i // horizontal, i % horizontal]
        else:
            ax = axs[i % horizontal]
        adata.obs["bicluster"] = "not in cluster"
        cluster_cell = np.array(adata.obs_names)[bicluster.cell_index]
        adata.obs.loc[cluster_cell, "bicluster"] = "in cluster"
        sc.pl.umap(adata, color="bicluster", ax=ax, show=False, use_raw=False)
        ax.set_title(f"Bicluster {i}")
        ax.set_aspect('equal')
        ax.axis('off')

    while True:
        i += 1
        if i >= vertical * horizontal:
            break
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    
def plot_biclusters_umap_array(adata: sc.AnnData, biclusters, save_path=None):
    """
    Plot UMAP plot grouped by each bicluster. 
    """
    if len(biclusters) == 0:
        raise ValueError("No bicluster is provided.")
    # If no UMAP is computed, compute it
    if "X_umap" not in adata.obsm.keys():
        sc.pp.neighbors(adata, use_rep="X")
        sc.tl.umap(adata)
    
    # Plot UMAP plot for each bicluster
    vertical = int(np.sqrt(len(biclusters)))
    horizontal = int(np.ceil(len(biclusters) / vertical))
    fig, axs = plt.subplots(vertical, horizontal, figsize=(30, 30))
    for i, bicluster in enumerate(biclusters):
        if axs.ndim == 2:
            ax = axs[i // horizontal, i % horizontal]
        else:
            ax = axs[i % horizontal]
        adata.obs["bicluster"] = "not in cluster"
        cluster_cell = np.array(adata.obs_names)[bicluster.cell_index]
        adata.obs.loc[cluster_cell, "bicluster"] = "in cluster"
        sc.pl.umap(adata, color="bicluster", ax=ax, show=False, use_raw=False)
        ax.set_title(f"Bicluster {bicluster.cluster_num}")
        ax.set_aspect('equal')
        ax.axis('off')

    while True:
        i += 1
        if i >= vertical * horizontal:
            break
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

# def plot_biclusters_heatmap(adata: sc.AnnData, biclusters: list, save_path=None):
#     """
#     Plot matshow for each bicluster and outline the biclustered area with a square.
#     """
#     if len(biclusters) == 0:
#         raise ValueError("No bicluster is provided.")
    
#     vertical = int(np.sqrt(len(biclusters)))
#     horizontal = int(np.ceil(len(biclusters) / vertical))
#     fig, axs = plt.subplots(vertical, horizontal, figsize=(30, 30))
    
#     for i, bicluster in enumerate(biclusters):
#         if axs.ndim == 2:
#             ax = axs[i // horizontal, i % horizontal]
#         else:
#             ax = axs[i % horizontal]
#         row_comp = np.ones(adata.shape[0])
#         col_comp = np.ones(adata.shape[1])
#         row_comp[bicluster.cell_index] = 0
#         col_comp[bicluster.gene_index] = 0
        
#         mat = adata.X
#         if isinstance(mat, np.matrix):
#             mat = mat.todense()
#         elif scipy.sparse.isspmatrix_csr(mat):
#             mat = mat.toarray()
#         else:
#             mat = np.array(mat)
        
#         if sum(row_comp) == len(row_comp) or sum(col_comp) == len(col_comp):
#             continue
#         mat = mat[np.argsort(row_comp), :][:, np.argsort(col_comp)]
#         mat = np.array(mat, dtype=float)

#         vmax = np.median(mat)
#         im = ax.matshow(mat, vmin=0, vmax=vmax, cmap="viridis")
#         fig.colorbar(im, ax=ax)
        
#         rect = plt.Rectangle((0, 0), len(bicluster.gene_index), len(bicluster.cell_index),
#                              fill=False, edgecolor="red", lw=3)
#         ax.add_patch(rect)
#         ax.set_title(f"Bicluster {i}")
#         ax.set_aspect('equal')
    
#     while True:
#         i += 1
#         if i >= vertical * horizontal:
#             break
#         fig.delaxes(axs.flatten()[i])
    
#     # Optionally save the figure if save_path is provided
#     if save_path is not None:
#         plt.savefig(save_path)
#     plt.show()

def plot_biclusters_heatmap(adata: sc.AnnData, biclusters: list, save_path=None):
    """
    Plot matshow for each bicluster and outline the biclustered area with a square.
    With a single shared colorbar and centered last row.
    """
    plt.clf()
    if len(biclusters) == 0:
        raise ValueError("No bicluster is provided.")
    
    ncols = min(4, len(biclusters))
    nrows = int(np.ceil(len(biclusters) / ncols))
    
    last_row_count = len(biclusters) - (nrows - 1) * ncols
    
    fig = plt.figure(figsize=(4*ncols, 4*nrows))
    
    mat = adata.X.copy() if hasattr(adata.X, 'copy') else np.array(adata.X)
    if scipy.sparse.issparse(mat):
        mat = mat.toarray()
    elif isinstance(mat, np.matrix):
        mat = np.array(mat)
    
    global_vmin = 0
    global_vmax = np.percentile(mat.flatten(), 95)

    gs = fig.add_gridspec(nrows, ncols+1, 
                          width_ratios=[1]*ncols + [0.05],
                          hspace=0.4,
                          wspace=0.2)
    
    all_im = []
    
    for i, bicluster in enumerate(biclusters):
        row_idx = i // ncols
        
        if row_idx == nrows - 1:
            offset = (ncols - last_row_count) // 2
            last_row_position = i - (nrows - 1) * ncols
            col_idx = last_row_position + offset
        else:
            col_idx = i % ncols
        
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        row_indices = np.zeros(adata.shape[0], dtype=bool)
        col_indices = np.zeros(adata.shape[1], dtype=bool)
        row_indices[bicluster.cell_index] = True
        col_indices[bicluster.gene_index] = True
        
        reordered_rows = np.concatenate([np.where(row_indices)[0], np.where(~row_indices)[0]])
        reordered_cols = np.concatenate([np.where(col_indices)[0], np.where(~col_indices)[0]])
        
        plot_mat = mat[reordered_rows, :][:, reordered_cols]
        
        im = ax.matshow(plot_mat, vmin=float(global_vmin), vmax=float(global_vmax), cmap="viridis")
        all_im.append(im)

        rect = plt.Rectangle((0, 0), 
                            len(bicluster.gene_index), 
                            len(bicluster.cell_index),
                            fill=False, edgecolor="red", lw=2)
        ax.add_patch(rect)

        ax.set_title(f"Bicluster {i}", fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        
    cbar_ax = fig.add_subplot(gs[:, -1])
    if all_im:
        cbar = fig.colorbar(all_im[0], cax=cbar_ax)
        cbar.set_ticks(np.arange(float(global_vmin), float(global_vmax)))
        cbar.set_label('Expression Value', fontsize=18)

    # plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight') 
    plt.show()

def _process_assignments_from_column(data_series: pd.Series):
    """
    Processes a pandas Series to extract assignments for each item (cell).
    Handles single values, lists, NumPy arrays, and string representations of lists.
    All assignment values are converted to strings for consistency.

    Parameters:
    -----------
    data_series : pd.Series
        The Series from adata.obs[key] containing assignment data.

    Returns:
    --------
    assignments_per_item : list[list[str]]
        A list where each inner list contains string assignments for the corresponding item.
    unique_values_sorted : list[str]
        A sorted list of unique string assignment values found.
    """
    assignments_per_item = []
    all_values_flat = set()

    for entry in data_series:
        current_item_assignments = []
        
        # MINIMAL CHANGE APPLIED HERE:
        # Prioritize checking for list or NumPy array instances first.
        # This ensures that if 'entry' is an array, it's processed by iterating
        # its elements, and 'pd.isna(entry)' is not called directly on the array
        # in a way that would produce a boolean array for the 'if' condition.
        if isinstance(entry, (list, np.ndarray)): 
            # The list comprehension correctly handles iterating through elements 
            # of both Python lists and 1D NumPy arrays, applying pd.isna to each element.
            current_item_assignments = [str(x) for x in entry if not pd.isna(x)]
        elif pd.isna(entry): # This will now primarily handle scalar NA types (None, np.nan, pd.NA)
                             # as lists/arrays are caught by the condition above.
            pass # No assignments for this item if it's a scalar NA
        elif isinstance(entry, str):
            try:
                # Attempt to parse string as a Python literal (e.g., "[1, 2]", "['a', 'b']")
                evaluated_entry = ast.literal_eval(entry)
                if isinstance(evaluated_entry, list):
                    current_item_assignments = [str(x) for x in evaluated_entry if not pd.isna(x)]
                else:
                    # literal_eval succeeded but resulted in a scalar (e.g. string "'item1'")
                    if not pd.isna(evaluated_entry): # Check if the scalar result is NA
                        current_item_assignments = [str(evaluated_entry)]
            except (ValueError, SyntaxError, TypeError):
                # If literal_eval fails, treat the string as a single category
                current_item_assignments = [entry] # entry is already a string
        else:
            # For numbers or other scalar types not covered above
            if not pd.isna(entry): # Check if the scalar entry is NA
                current_item_assignments = [str(entry)]
        
        assignments_per_item.append(current_item_assignments)
        for val in current_item_assignments:
            all_values_flat.add(val)
            
    unique_values_sorted = sorted(list(all_values_flat))
    return assignments_per_item, unique_values_sorted

def create_cell_sankey_diagram(adata, source_key, target_key,
                                       title="Distribution: Source to Target",
                                       save_path=None):
    """
    Create a Sankey diagram showing relationships between two annotation keys 
    from an AnnData object. Both source and target keys can have single or 
    multiple assignments per cell (e.g. simple strings, lists, or string-encoded lists).

    Parameters:
    -----------
    adata : AnnData
        AnnData object with cell data.
    source_key : str
        Key in adata.obs for source categories.
    target_key : str
        Key in adata.obs for target categories.
    title : str
        Title for the Sankey diagram.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
        Supported formats: .html, .png, .jpg, .svg, .pdf (requires kaleido).

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure object with the Sankey diagram.
    """
    print(f"Processing {len(adata.obs_names)} items (cells/observations)...")

    print(f"Extracting assignments for source key: '{source_key}'...")
    source_assignments_per_item, unique_sources = _process_assignments_from_column(adata.obs[source_key])

    print(f"Extracting assignments for target key: '{target_key}'...")
    target_assignments_per_item, unique_targets = _process_assignments_from_column(adata.obs[target_key])

    print(f"Found {len(unique_sources)} unique source categories and {len(unique_targets)} unique target categories.")

    source_map = {val: i for i, val in enumerate(unique_sources)}
    target_map = {val: i + len(unique_sources) for i, val in enumerate(unique_targets)}

    source_node_labels = [f"{source_key}: {s}" for s in unique_sources]
    target_node_labels = [f"{target_key}: {t}" for t in unique_targets]
    all_node_labels = source_node_labels + target_node_labels

    print("Calculating flows...")
    flow_counts = {}
    for i in range(len(adata.obs_names)):
        current_item_sources = source_assignments_per_item[i]
        current_item_targets = target_assignments_per_item[i]

        for s_val in current_item_sources:
            s_idx = source_map[s_val] 
            for t_val in current_item_targets:
                t_idx = target_map[t_val]
                
                link_key = (s_idx, t_idx)
                flow_counts[link_key] = flow_counts.get(link_key, 0) + 1
    
    sankey_sources = []
    sankey_targets = []
    sankey_values = []
    
    for (s_idx, t_idx), count in flow_counts.items():
        sankey_sources.append(s_idx)
        sankey_targets.append(t_idx)
        sankey_values.append(count)

    source_node_colors = generate_distinct_colors_str(len(unique_sources))
    
    node_colors_plotly = []
    for i in range(len(all_node_labels)):
        if i < len(unique_sources):
            node_colors_plotly.append(source_node_colors[i])
        else:
            node_colors_plotly.append('rgba(200, 200, 200, 0.8)')

    link_colors_plotly = [source_node_colors[s_idx] for s_idx in sankey_sources]

    # Create Sankey diagram
    print("Creating Sankey diagram...")
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,               # Vertical padding between nodes
            thickness=20,         # Thickness of the nodes
            line=dict(color="black", width=0.5),
            label=all_node_labels,
            color=node_colors_plotly
        ),
        link=dict(
            source=sankey_sources,
            target=sankey_targets,
            value=sankey_values,
            color=link_colors_plotly
        )
    ))

    fig.update_layout(
        title_text=title,
        font=dict(size=12, color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=max(600, max(len(unique_sources), len(unique_targets)) * 25 + 150), 
        margin=dict(l=50, r=50, t=60, b=40)
    )

    if save_path:
        print(f"Saving diagram to {save_path}...")
        try:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                # For static images like png, jpg, svg, pdf, kaleido needs to be installed
                # pip install -U kaleido
                fig.write_image(save_path)
            print(f"Successfully saved to {save_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
            print("Please ensure 'kaleido' is installed for static image formats (pip install -U kaleido).")

    return fig

def generate_distinct_colors_str(n):
    """Generate visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + 0.3 * (i % 3) / 2
        lightness = 0.4 + 0.2 * ((i // 3) % 3) / 2
        
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)')
    
    return colors


def generate_distinct_colors_tuple(n):
    """Generate visually distinct colors."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + 0.3 * (i % 3) / 2
        lightness = 0.4 + 0.2 * ((i // 3) % 3) / 2
        
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        # Return as tuple instead of rgba string
        colors.append((r, g, b, 0.8))
    
    return colors

def plot_clusters_convex(x, y, ax, base_color, epsilon=0.2, min_samples=5, label=None):
    """
    Plot subclusters using DBSCAN on 2D data.
    """
    points = np.column_stack([x, y])
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(points)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if isinstance(base_color, tuple):
        base_rgb = np.array(base_color[:3])  # Convert to numpy array for element-wise multiplication
    else:
        # If it's a string color name, convert to RGB
        base_rgb = np.array(plt.cm.colors.to_rgb(base_color))
    
    # Generate color variations
    color_variations = [tuple(base_rgb * (0.5 + 0.5 * i/max(1, n_clusters))) for i in range(n_clusters)]
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        if sum(cluster_mask) < 3:
            continue     
        cluster_points = points[cluster_mask]
        try:
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 
                        color=base_color if isinstance(base_color, str) else base_color[:3], 
                        linewidth=2)
            hull_pts = cluster_points[hull.vertices]
            color = color_variations[cluster_id % len(color_variations)]
            ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=0.2)
        except Exception as e:
            # For debugging, print the actual exception
            print(f"Error in hull calculation: {e}")
            pass
        
def plot_biclusters_convex_umap(adata: sc.AnnData, biclusters, save_path=None, title=None):
    """
    Plot convex hulls for each bicluster on UMAP coordinates using DBSCAN.
    """
    if "X_umap" not in adata.obsm:
        print("No X_umap provided in adata.obsm, Computing UMAP...")
        sc.pp.neighbors(adata, use_rep="X")
        sc.tl.umap(adata)

    x = adata.obsm["X_umap"][:, 0]
    y = adata.obsm["X_umap"][:, 1]

    colors = generate_distinct_colors_tuple(len(biclusters))

    fig, ax = plt.subplots(figsize=(8, 8))
    # First plot background points
    ax.scatter(x, y, s=5, color='lightgray', alpha=0.2)
    
    for i, bc in enumerate(biclusters):
        cluster_cells = bc.cell_index
        x_bc = x[cluster_cells]
        y_bc = y[cluster_cells]
        # Plot the points in this cluster first
        ax.scatter(x_bc, y_bc, s=10, color=colors[i], alpha=0.5, label=f"Bicluster {i}")
        # Then plot the convex hulls
        plot_clusters_convex(x_bc, y_bc, ax, colors[i], epsilon=0.3, min_samples=5)
    
    ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title, fontsize=14)
    else:
        pass
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.legend()
    
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()