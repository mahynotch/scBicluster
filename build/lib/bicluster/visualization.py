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

def create_cell_sankey_diagram(adata, source_key="louvain", target_key="biclusters", 
                              title="Cell Distribution: Louvain Clusters to Biclusters",
                              save_path=None):
    """
    Create a Sankey diagram showing relationships between louvain clusters and biclusters
    when cells can belong to multiple biclusters.
    
    Parameters:
    -----------
    adata : AnnData
        AnnData object with cell data
    source_key : str
        Key in adata.obs for louvain clusters or other data (single membership)
    target_key
 : str
        Key in adata.obs for bicluster assignments (multiple memberships in lists)
    title : str
        Title for the Sankey diagram
    save_path : str, optional
        Path to save the figure, if None figure is not saved
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Plotly figure with the Sankey diagram
    """
    print(f"Processing {len(adata.obs_names)} cells...")
    
    cells = list(adata.obs_names)
    
    louvain_values = adata.obs[source_key].values
    louvain_map = {} 
    cell_to_louvain = {}

    unique_louvains = sorted(set(louvain_values))
    for i, lv in enumerate(unique_louvains):
        # louvain_name = f"Louvain {lv}"
        louvain_map[lv] = i
    
    for cell, lv in zip(cells, louvain_values):
        cell_to_louvain[cell] = lv
    
    all_biclusters = set()
    cell_to_biclusters = {}
    
    print("Extracting biclusters...")
    for cell, bc_entry in zip(cells, adata.obs[target_key
]):
        if isinstance(bc_entry, list):
            biclusters_for_cell = bc_entry
        elif isinstance(bc_entry, str):
            try:
                biclusters_for_cell = eval(bc_entry)
                if not isinstance(biclusters_for_cell, list):
                    biclusters_for_cell = [biclusters_for_cell]
            except:
                biclusters_for_cell = [x.strip() for x in bc_entry.strip('[]').split(',') if x.strip()]
                try:
                    biclusters_for_cell = [int(x) for x in biclusters_for_cell]
                except:
                    pass
        else:
            biclusters_for_cell = [bc_entry]

        all_biclusters.update(biclusters_for_cell)

        cell_to_biclusters[cell] = biclusters_for_cell

    bicluster_list = sorted(all_biclusters)
    bicluster_map = {bc: i + len(unique_louvains) for i, bc in enumerate(bicluster_list)}
    print(f"Found {len(unique_louvains)} in source and {len(bicluster_list)} in target.")
    nodes = [f"{source_key} {lv}" for lv in unique_louvains] + [f"{target_key} {bc}" for bc in bicluster_list]
    
    louvain_colors = generate_distinct_colors_str(len(unique_louvains))
    print("Calculating flows...")
    flow_counts = {} 
    
    for cell in cells:
        louvain = cell_to_louvain[cell]
        biclusters = cell_to_biclusters[cell]
        
        louvain_idx = louvain_map[louvain]
        
        for bc in biclusters:
            bc_idx = bicluster_map[bc]
            key = (louvain_idx, bc_idx)
            
            if key not in flow_counts:
                flow_counts[key] = 0
            flow_counts[key] += 1
  
    sources = []
    targets = []
    values = []
    link_colors = []
    
    for (source, target), value in flow_counts.items():
        sources.append(source)
        targets.append(target)
        values.append(value)
        link_colors.append(louvain_colors[source])
    
    node_colors = []
    for i in range(len(nodes)):
        if i < len(unique_louvains):
            node_colors.append(louvain_colors[i])
        else:
            node_colors.append('rgba(180, 180, 180, 0.8)')
    
    print("Creating Sankey diagram...")
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors
        )
    ))
    
    fig.update_layout(
        title_text=title,
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        margin=dict(l=25, r=25, t=40, b=25)
    )
    
    if save_path:
        print(f"Saving diagram to {save_path}")
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
    
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