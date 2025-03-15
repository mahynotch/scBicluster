import pandas as pd
import numpy as np
from typing import List, Tuple, Literal
# import scanpy as sc
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np
import scanpy
from .algorithms import count_incluster_fold_change
from .utils import CustomBiclusterClass, BiclusterList

class GESA:
    def __init__(self, adata, gene_set: List[str], expression_key: str = 'X', weight: float = 1.0):
        """
        Initialize GESA with an AnnData object and gene set.

        Args:
            adata: AnnData object containing single-cell RNA-seq data.
            gene_set: List of genes in the pathway or set of interest.
            expression_key: The key in adata.layers to use for expression data (or fallback to adata.X).
            weight: Exponent for weighting gene expression values.
        """
        self.adata = adata
        # Restrict gene_set to genes present in adata.var_names.
        self.gene_set = set(gene for gene in gene_set if gene in adata.var_names)
        self.expression_key = expression_key
        self.weight = weight

    def get_expression_vector(self, cell: str) -> Tuple[List[str], np.ndarray]:
        """
        Given a cell identifier, return a tuple (gene_names, expr_values) for that cell.
        It first attempts to use adata.layers[expression_key] and falls back to adata.X if needed.
        """
        if self.expression_key in self.adata.layers:
            expr_mat = self.adata[cell, :].layers[self.expression_key]
            # Convert sparse matrix to dense if needed.
            if hasattr(expr_mat, "toarray"):
                expr_values = expr_mat.toarray().flatten()
            else:
                expr_values = np.array(expr_mat).flatten()
        else:
            expr = self.adata[cell, :].X
            if hasattr(expr, "toarray"):
                expr_values = expr.toarray().flatten()
            else:
                expr_values = np.array(expr).flatten()
        gene_names = list(self.adata.var_names)
        return gene_names, expr_values

    def rank_genes(self, cell: str) -> List[Tuple[str, float]]:
        """
        Rank genes in descending order of expression for the given cell.

        Returns:
            A list of tuples (gene, expression_value), descending sorted.
        """
        gene_names, expr_values = self.get_expression_vector(cell)
        gene_expression = list(zip(gene_names, expr_values))
        ranked_genes = sorted(gene_expression, key=lambda x: x[1], reverse=True)
        return ranked_genes

    def calculate_enrichment_score(
        self, cell: str, return_ranking: bool = False
    ) -> Tuple[float, List[float], Optional[List[Tuple[str, float]]]]:
        """
        Calculate the enrichment score for a given cell using a weighted running-sum statistic.

        Args:
            cell: Cell identifier.
            return_ranking: If True, also return the ranked gene list.

        Returns:
            A tuple containing:
              - final enrichment score (float),
              - list of running enrichment scores (List[float]),
              - ranked genes (List[Tuple[str, float]]) if return_ranking is True, else None.
        """
        ranked_genes = self.rank_genes(cell)
        N = len(ranked_genes)
        N_R = len(self.gene_set)
        if N_R == 0 or N_R == N:
            raise ValueError("Gene set is empty or contains all genes, cannot compute enrichment.")

        # Compute weights and normalization for genes in gene_set
        sum_weights = 0.0
        gene_weight = {}
        for gene, expr in ranked_genes:
            if gene in self.gene_set:
                w = np.abs(expr) ** self.weight
                gene_weight[gene] = w
                sum_weights += w

        running_score = 0.0
        running_scores = []
        for i, (gene, expr) in enumerate(ranked_genes):
            if gene in self.gene_set:
                increment = gene_weight[gene] / sum_weights if sum_weights > 0 else 0
                running_score += increment
            else:
                running_score -= 1 / (N - N_R)
            running_scores.append(running_score)

        enrichment_score = max(running_scores, key=abs)
        ranked_out = ranked_genes if return_ranking else None
        return enrichment_score, running_scores, ranked_out

    def calculate_significance(self, cell: str, n_permutations: int = 1000) -> float:
        """
        Calculate the statistical significance (p-value) of the observed enrichment score using a permutation test.

        Args:
            cell: Cell identifier.
            n_permutations: Number of permutations to perform.

        Returns:
            p-value representing the fraction of permutations that yield an enrichment score 
            with an absolute value equal to or greater than that of the actual cell.
        """
        actual_score, _, _ = self.calculate_enrichment_score(cell)
        gene_names, expr_values = self.get_expression_vector(cell)
        perm_scores = []

        for _ in range(n_permutations):
            shuffled = expr_values.copy()
            np.random.shuffle(shuffled)
            # Build permuted ranking by zipping gene_names with shuffled expression.
            perm_gene_expression = list(zip(gene_names, shuffled))
            perm_ranked = sorted(perm_gene_expression, key=lambda x: x[1], reverse=True)
            
            N = len(perm_ranked)
            N_R = len(self.gene_set)
            sum_weights = sum(
                (np.abs(expr) ** self.weight)
                for gene, expr in perm_ranked
                if gene in self.gene_set
            )
            running_score = 0.0
            max_score = 0.0
            for i, (gene, expr) in enumerate(perm_ranked):
                if gene in self.gene_set:
                    inc = (np.abs(expr) ** self.weight) / sum_weights if sum_weights > 0 else 0
                    running_score += inc
                else:
                    running_score -= 1 / (N - N_R)
                if abs(running_score) > abs(max_score):
                    max_score = running_score
            perm_scores.append(max_score)

        p_value = sum(1 for score in perm_scores if abs(score) >= abs(actual_score)) / n_permutations
        return p_value
    
    
class AnnotationDB:
    def __init__(self, db_source: Literal["PanglaoDB", "cellmarkerdb", "DataFrame"], db_path):
        """
        Initialize the annotation database.

        Parameters
        ----------
        db_source : Literal["PanglaoDB", "cellmarkerdb", "DataFrame"]
            The source type. If "DataFrame", db_path is assumed to be an actual DataFrame.
        db_path : str or pd.DataFrame
            File path to the database file or a DataFrame.
        """
        if db_source == "PanglaoDB":
            db = pd.read_csv(db_path, sep='\t', index_col=0)
            # Here we assume that the database file has columns: "official gene symbol" and "cell type"
            self.db = pd.DataFrame({
                "marker_gene": db["official gene symbol"],
                "cell_type": db["cell type"]
            }).set_index("marker_gene")
        elif db_source == "cellmarkerdb":
            db = pd.read_excel(db_path, index_col=0)
            # Assume the Excel file has columns: "marker" and "cell_name"
            self.db = pd.DataFrame({
                "marker_gene": db["marker"],
                "cell_type": db["cell_name"]
            }).set_index("marker_gene")
        else:
            # If a DataFrame is provided directly
            df = db_path.copy()
            if "marker_gene" not in df.columns or "cell_type" not in df.columns:
                raise ValueError("DataFrame must contain columns 'marker_gene' and 'cell_type'.")
            self.db = df.set_index("marker_gene")

    def compute_likelihood(self, gene_list: List[str]) -> pd.Series:
        """
        Compute a likelihood score for each cell type based on marker gene counts.

        Parameters
        ----------
        gene_list : List[str]
            List of gene names (e.g., the top-ranked genes for a cell).

        Returns
        -------
        pd.Series
            A series with cell types as index and vote counts (likelihoods) as values.
        """
        votes = defaultdict(int)
        for gene in gene_list:
            match = self.db.index.str.fullmatch(gene, case=False)
            if match.any():
                query_gene = self.db.index[match][0]
            else:
                continue
            cell_type = self.db.loc[query_gene, "cell_type"]
            # In case one gene is assigned to multiple cell types (if stored as a Series),
            # count each occurrence.
            if isinstance(cell_type, pd.Series):
                for ct in cell_type:
                    votes[ct] += 1
            else:
                votes[cell_type] += 1
        return pd.Series(votes)

    def most_likely_cell_type(self, gene_list: List[str]) -> str:
        """
        Determine the most likely cell type based on marker gene counts.

        Parameters
        ----------
        gene_list : List[str]
            A list of gene names (e.g., from the top-ranked genes of a cell).

        Returns
        -------
        str
            The cell type with the highest vote count. Returns "Unknown" if no marker is present.
        """
        likelihood = self.compute_likelihood(gene_list)
        likelihood = likelihood / likelihood.sum()
        return likelihood.nlargest(5)
    
    def annotate_anndata_auto(self, adata, bicluster: CustomBiclusterClass, top_gene_num: int, expression_key: str = 'X') -> pd.DataFrame:
        """
        Automatically annotate each cell in an AnnData object using marker gene information from the database.
        For each cell, the top_n expressed genes are used to vote on the cell type.

        Parameters
        ----------
        adata : AnnData
            An AnnData object containing single-cell RNA-seq data.
        expression_key : str
            The key in adata.layers (if available) or in adata.X to use for the expression values.
        top_n : int
            Number of top-ranked genes (by expression) to use for marker voting.

        Returns
        -------
        pd.DataFrame
            A DataFrame with cell IDs as the index and columns for the most likely cell type.
        """
        if expression_key in adata.layers:
            expr_data = adata.layers[expression_key]
            # Convert to a dense array if needed.
            if hasattr(expr_data, "toarray"):
                values = expr_data.toarray()
            else:
                values = np.array(expr_data)
        else:
            expr_data = adata.X
            if hasattr(expr_data, "toarray"):
                values = expr_data.toarray()
            else:
                values = np.array(expr_data)
        values = count_incluster_fold_change(adata, bicluster)
        gene_names = adata[:, bicluster.gene_index].var_names
        if gene_names[0].startswith("ENSG"):
            gene_names = adata[:, bicluster.gene_index].var["feature_name"]
        # Generate ranking of genes (descending order by expression).
        ranked_genes = sorted(zip(gene_names, values), key=lambda x: x[1], reverse=True)
        # Select the top_n genes.
        print(ranked_genes)
        top_genes = []
        for gene, expr in ranked_genes[:top_gene_num]:
            if expr > 1:
                top_genes.append(gene)
        top_genes = set(list(map(lambda x: x.split(".")[0], top_genes)))
        cell_type = self.most_likely_cell_type(top_genes)
        return cell_type
