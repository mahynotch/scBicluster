import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import spearmanr
from sklearn.model_selection import train_test_split, cross_val_score
from .utils import CustomBiclusterClass, BiclusterList
from .algorithms import count_all_fold_change, count_incluster_fold_change
from typing import List, Set, Tuple, Optional

class GSEAConfidenceClassifier:
    """
    A classifier that uses gene set (bicluster) enrichment scores as features.
    
    For each bicluster (module), the class assumes a CustomBicluster object. For each cell,
    it computes a confidence/enrichment score based on how highly the genes in that gene set
    are expressed relative to the full ranked expression vector.
    
    These scores are then used to train a classifier (default: Random Forest) to predict cell labels.
    
    Parameters:
        classifier: A classifier instance. It should be a object implemented fit and predict. If not provided,
            defaults to RandomForestClassifier.
        random_state (int): Random state for reproducibility.
    """
    
    def __init__(self, classifier=None, random_state=42):
        self.random_state = random_state
        if classifier is None:
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            self.classifier = classifier
            
    def _generate_ranked_gene_list(self, adata: sc.AnnData, gene_index: np.array, fc: np.array) -> Tuple[np.array, np.array]:
        genes = adata.var_names[gene_index]
        fc_subset = fc[gene_index]
        sorted_idx = np.argsort(-fc_subset)
        return list(genes[sorted_idx]), fc_subset[sorted_idx]

    # def annotate_single_cell(
    #     self,
    #     gene_names: List[str],
    #     expr_values: np.array,
    #     upgenes,
    #     downgenes,
    #     return_label: bool = True
    #     ) -> Tuple[float, Optional[str]]:
    #     """
    #     Annotate a single cell based on up/down gene sets using rank-based correlation (Spearman).
        
    #     Parameters
    #     ----------
    #     gene_names : List[str]
    #         A list of gene names for this cell (unsorted).
    #     expr_values : List[float]
    #         A list of gene expression values corresponding to gene_names (unsorted).
    #     return_label : bool, optional (default=True)
    #         If True, also return a simple annotation label ("up-regulated" or "down-regulated")
    #         based on the sign of the correlation.
        
    #     Returns
    #     -------
    #     score : float
    #         Spearman correlation between the cell's expression ranks and the +1/-1 signature.
    #     label : str or None
    #         "up-regulated" if score > 0, "down-regulated" if score < 0, and "neutral" if score = 0.
    #         Returned only if return_label is True, otherwise None.
        
    #     Notes
    #     -----
    #     - A positive score means the cell's top-ranked genes overlap more with up_genes.
    #     - A negative score means the cell's top-ranked genes overlap more with down_genes.
    #     - Genes not in (up_genes ∪ down_genes) get signature = 0, which won't affect the correlation.
    #     """
    #     up_genes = set(upgenes)
    #     down_genes = set(downgenes)
        
    #     if len(gene_names) != len(expr_values):
    #         raise ValueError("gene_names and expr_values must have the same length.")
        
    #     # 1) Build a signature vector for each gene: +1 (up), -1 (down), or 0
    #     signature = np.zeros(len(gene_names))
    #     for i, g in enumerate(gene_names):
    #         if g in up_genes:
    #             signature[i] = 1.0
    #         elif g in down_genes:
    #             signature[i] = -1.0
    #         else:
    #             # signature[i] = 0.0
    #             pass
    #     signature = np.array(signature, dtype=float)
        
    #     expr_series = pd.Series(expr_values)
    #     ranks = expr_series.rank(method="average", ascending=True).values 
    #     rho, _ = spearmanr(ranks, signature, nan_policy="omit")
        
    #     label = None
    #     if return_label:
    #         if rho > 0:
    #             label = "up-regulated"
    #         elif rho < 0:
    #             label = "down-regulated"
    #         else:
    #             label = "neutral"
        
    #     return rho, label
    
    def annotate_single_cell(
        self,
        gene_names: List[str],
        expr_values: np.array,
        upgenes,
        downgenes,
        normalize: bool = True
        ) -> Tuple[float, Optional[str]]:
        """
        Annotate a single cell based on up/down gene sets using rank-based correlation (Spearman).
        
        Parameters
        ----------
        gene_names : List[str]
            A list of gene names for this cell (unsorted).
        expr_values : List[float]
            A list of gene expression values corresponding to gene_names (unsorted).
        return_label : bool, optional (default=True)
            If True, also return a simple annotation label ("up-regulated" or "down-regulated")
            based on the sign of the correlation.
        
        Returns
        -------
        score : float
            Spearman correlation between the cell's expression ranks and the +1/-1 signature.
        label : str or None
            "up-regulated" if score > 0, "down-regulated" if score < 0, and "neutral" if score = 0.
            Returned only if return_label is True, otherwise None.
        
        Notes
        -----
        - A positive score means the cell's top-ranked genes overlap more with up_genes.
        - A negative score means the cell's top-ranked genes overlap more with down_genes.
        - Genes not in (up_genes ∪ down_genes) get signature = 0, which won't affect the correlation.
        """
        
        if len(gene_names) != len(expr_values):
            raise ValueError("gene_names and expr_values must have the same length.")
        
        ranked_genes = np.array(gene_names, dtype=str)
        ranked_genes = ranked_genes[np.argsort(-expr_values)]
        
        feature_length = len(upgenes)
        
        ES = np.zeros(feature_length)
        max_ES = np.zeros(feature_length)
        num_up_inter = np.zeros(feature_length)
        num_down_inter = np.zeros(feature_length)
        
        for i in range(feature_length):
            num_up_inter[i] = len(np.intersect1d(ranked_genes, upgenes[i]))
            num_down_inter[i] = len(np.intersect1d(ranked_genes, downgenes[i]))
        
        for i in range(len(ranked_genes)):
            curr_gene = ranked_genes[i]
            for j in range(feature_length):
                if curr_gene in upgenes[j]:
                    ES[j] += 1 / num_up_inter[j]
                elif curr_gene in downgenes[j]:
                    ES[j] -= 1 / num_down_inter[j]
                if ES[j] > max_ES[j]:
                    max_ES[j] = ES[j]
        
        # if max_ES > 0:
        #     label = "up-regulated"
        # elif max_ES < 0:
        #     label = "down-regulated"
        # else:
        #     label = "neutral"
        # if return_label:
        #     return max_ES, label
        # else:
        if normalize == True:
            max_ES = (max_ES - np.std(ES)) / (np.mean(max_ES) + 1e-10)
        return max_ES.flatten(), None
        
        
        

    def _get_enrichment_features(self, adata: sc.AnnData):
        """
        Computes an enrichment feature matrix for all cells.
        
        Each row corresponds to a cell and each column to a bicluster's enrichment score.
        
        Parameters:
            adata (AnnData): AnnData object that contains gene expression data.
            biclusters (list): List of dictionaries with the key "gene_list".
            
        Returns:
            enrichment_scores (numpy.array): Matrix of shape (n_cells, n_biclusters)
        """
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        feature_matrix = np.zeros((adata.n_obs, len(self.biclusters)))
        means = np.mean(X, axis=0)
        vars = np.var(X, axis=0)
        for i, cell_expression in enumerate(X):
            # for j in range(len(self.biclusters)):        
            score = self.annotate_single_cell(adata.var_names, cell_expression, self.positive_genes, self.negative_genes)[0]
            feature_matrix[i, :] = score
        return feature_matrix

    def fit(self, adata, biclusters: BiclusterList, labels_key='cell_type', one_hot=False):
        """
        Computes the enrichment features from the AnnData object and biclusters, and
        trains the classifier on these features.
        
        Parameters:
            adata (AnnData): AnnData object with gene expression data.
            biclusters (list): List of dictionaries with key "gene_list".
            labels_key (str): Key in adata.obs that contains cell labels.
            test_size (float): Fraction of data to use as a test set for internal evaluation.
        
        Returns:
            score (float): Average 5-fold cross-validation score.
        """
        self.adata = adata
        self.biclusters = biclusters
        self.labels_key = labels_key
        self.one_hot = one_hot
        cell_types = adata.obs[labels_key]
        if one_hot:
            y = pd.get_dummies(cell_types)
        else:
            y = cell_types
        biclusters_positive_genesets = [None] * len(biclusters)
        biclusters_positive_score = [None] * len(biclusters)
        biclusters_negative_genesets = [None] * len(biclusters)
        biclusters_negative_score = [None] * len(biclusters)
        for idx, bicluster in enumerate(biclusters):
            fc = count_all_fold_change(adata, bicluster)
            biclusters_positive_genesets[idx], biclusters_positive_score[idx] = self._generate_ranked_gene_list(adata, bicluster.gene_index, fc)            
            # biclusters_negative_genesets[idx] = adata.var_names[np.setdiff1d(np.arange(adata.n_vars), bicluster.gene_index)]
            # biclusters_negative_score[idx] = fc[np.setdiff1d(np.arange(adata.n_vars), bicluster.gene_index)]
            biclusters_negative_genesets[idx], biclusters_negative_score[idx] = self._generate_ranked_gene_list(adata, np.setdiff1d(np.arange(adata.n_vars), bicluster.gene_index), fc)
        self.positive_genes = biclusters_positive_genesets
        self.positive_score = biclusters_positive_score
        self.negative_genes = biclusters_negative_genesets
        self.negative_score = biclusters_negative_score
        self.gene_list = np.union1d(np.concatenate(self.positive_genes), np.concatenate(self.negative_genes))
        self.cell_types = np.unique(cell_types)
        self.classifier.fit(self._get_enrichment_features(adata), y)

    def predict(self, adata, output_key='predicted_cell_type'):
        """
        Computes the enrichment features for new data and uses the trained classifier to predict labels.
        
        Parameters:
            adata (AnnData): AnnData object with gene expression data.
            biclusters (list): List of dictionaries with key "gene_index".
        
        Returns:
            predictions (numpy.array): Predicted cell type labels.
        """
        mask = np.isin(adata.var_names, self.gene_list)
        adata = adata[:, mask].copy()
        features = self._get_enrichment_features(adata)
        self.classifier.predict(features)
        if self.one_hot:
            adata.obs[output_key] = self.classifier.predict(features).argmax(axis=1)
        else:
            adata.obs[output_key] = self.classifier.predict(features) 

        return adata

    def transform(self, adata, biclusters):
        """
        Returns the enrichment feature matrix.
        
        Parameters:
            adata (AnnData): AnnData object with gene expression data.
            biclusters (list): List of dictionaries with key "gene_list".
        
        Returns:
            enrichment_features (numpy.array): Matrix of shape (n_cells, n_biclusters)
        """
        return self.get_enrichment_features(adata, biclusters)
    
    def get_gene_list(self):
        return self.gene_list
    
    def get_cell_types(self):
        return self.cell_types