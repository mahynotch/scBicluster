from biclustlib.algorithms import BitPatternBiclusteringAlgorithm
from biclustlib.algorithms import las, plaid, xmotifs, ModifiedChengChurchAlgorithm, ChengChurchAlgorithm
from biclustlib.algorithms.wrappers import IterativeSignatureAlgorithm2, RBinaryInclusionMaximalBiclusteringAlgorithm, RInClose, RPlaid, QualitativeBiclustering
from .algorithms import generate_random_biclusters
from.utils import BiclusterList, timed_execution, extract_mat
import scipy
from sklearn.cluster import SpectralBiclustering
import numpy as np
import scBC
import scBC.model

time_deco = timed_execution(timeout_seconds=60*60*5)


@time_deco
def cc_gen(adata, num_cluster=5):
    model = ChengChurchAlgorithm(num_cluster)
    mat = extract_mat(adata)
    biclustering = model.run(mat)
    biclusters = BiclusterList()
    for bicluster in biclustering.biclusters:
        biclusters.append(bicluster.rows, bicluster.cols)
    return biclusters

@time_deco
def scBC_gen(adata, max_iter=1000, num_cluster=5):
    model = scBC.model.scBC(adata=adata.copy(), layer="counts")
    model.train_VI(max_epochs=max_iter)
    model.get_reconst_data(n_samples=10)
    # model.get_edge()
    model.Biclustering(L=num_cluster)
    biclusters = BiclusterList()
    for cluster_num, bicluster in enumerate(model.S):
        biclusters.append(bicluster["subjects"], bicluster["measurements"])
    biclusters.merge(0.7)
    return biclusters

@time_deco
def spectral_gen(adata, num_cluster=5):
    model = SpectralBiclustering(n_clusters=num_cluster, method='log', n_init=10, random_state=0)
    mat = extract_mat(adata)
    model.fit(mat)
    biclusters = BiclusterList()
    for cluster_num in np.unique(model.row_labels_):
        row_index = np.where(model.row_labels_ == cluster_num)[0].flatten()
        col_index = np.where(model.column_labels_ == cluster_num)[0].flatten()
        biclusters.append(row_index, col_index)
    return biclusters

@time_deco
def bibit_gen(adata, minr=50, minc=50):
    model = BitPatternBiclusteringAlgorithm(minr, minc)
    mat = extract_mat(adata)
    biclustering = model.run(mat)
    biclusters = BiclusterList()
    for bicluster in biclustering.biclusters:
        biclusters.append(bicluster.rows, bicluster.cols)
    return biclusters

@time_deco
def xmotifs_gen(adata, num_cluster=5, alpha=0.05):
    model = xmotifs.ConservedGeneExpressionMotifs(num_cluster, alpha=alpha)
    mat = extract_mat(adata)
    biclustering = model.run(mat)
    biclusters = BiclusterList()
    for bicluster in biclustering.biclusters:
        biclusters.append(bicluster.rows, bicluster.cols)
    return biclusters

@time_deco
def las_gen(adata, num_cluster=5):
    model = las.LargeAverageSubmatrices(num_cluster)
    mat = extract_mat(adata)
    biclustering = model.run(mat)
    biclusters = BiclusterList()
    for bicluster in biclustering.biclusters:
        biclusters.append(bicluster.rows, bicluster.cols)
    return biclusters

@time_deco
def plaid_gen(adata, num_cluster=5):
    model = plaid.Plaid(num_cluster)
    mat = extract_mat(adata)
    biclustering = model.run(mat)
    biclusters = BiclusterList()
    for bicluster in biclustering.biclusters:
        biclusters.append(bicluster.rows, bicluster.cols)
    return biclusters

@time_deco
def isa_gen(adata, row_thr=2.0, col_thr=2.0):
    model = IterativeSignatureAlgorithm2(row_thr=row_thr, col_thr=col_thr)
    mat = extract_mat(adata)
    biclustering = model.run(mat)
    biclusters = BiclusterList()
    for bicluster in biclustering.biclusters:
        biclusters.append(bicluster.rows, bicluster.cols)
    return biclusters

@time_deco
def bimax_gen(adata, minr=50, minc=50, num_cluster=5):
    model = RBinaryInclusionMaximalBiclusteringAlgorithm(num_cluster, minr, minc)
    mat = extract_mat(adata)
    biclustering = model.run(mat)
    biclusters = BiclusterList()
    for bicluster in biclustering.biclusters:
        biclusters.append(bicluster.rows, bicluster.cols)
    return biclusters

@time_deco
def inclose_gen(adata, minr=50, minc=50, noise_tol=0.3):
    model = RInClose(minr, minc, noise_tol)
    mat = extract_mat(adata)
    biclustering = model.run(mat)
    biclusters = BiclusterList()
    for bicluster in biclustering.biclusters:
        biclusters.append(bicluster.rows, bicluster.cols)
    return biclusters

@time_deco
def qubic_gen(adata, num_cluster=5):
    model = QualitativeBiclustering(num_cluster)
    mat = extract_mat(adata)
    biclustering = model.run(mat)
    biclusters = BiclusterList()
    for bicluster in biclustering.biclusters:
        biclusters.append(bicluster.rows, bicluster.cols)
    return biclusters

@time_deco
def rplaid_gen(adata, num_cluster=5):
    model = RPlaid(num_cluster)
    mat = extract_mat(adata)
    biclustering = model.run(mat)
    biclusters = BiclusterList()
    for bicluster in biclustering.biclusters:
        biclusters.append(bicluster.rows, bicluster.cols)
    return biclusters

@time_deco
def rnd_bicluster_gen(adata, minr=50, minc=50, num_cluster=5):
    biclustering = generate_random_biclusters(adata, minr, minc, num_cluster)
    biclusters = BiclusterList()
    for rows, cols in biclustering:
        biclusters.append(rows, cols)
    return biclusters
