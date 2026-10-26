"""Exact grouped NMI from only observed truth/cluster contingency cells."""
import numpy as np
from scipy.sparse import csr_matrix
def prepare(group_codes,truth_codes,assignments,n_groups):
    group_codes=np.asarray(group_codes);truth_codes=np.asarray(truth_codes);seeds=[]
    n_truth=int(truth_codes.max())+1
    for assignment in assignments:
        cluster=np.unique(assignment,return_inverse=True)[1];n_cluster=int(cluster.max())+1
        pair,codes=np.unique(truth_codes*n_cluster+cluster,return_inverse=True)
        row=pair//n_cluster;col=pair%n_cluster;n_pair=len(pair)
        cells=csr_matrix((np.ones(len(codes)),(group_codes,codes)),shape=(n_groups,n_pair))
        truth=csr_matrix((np.ones(n_pair),(np.arange(n_pair),row)),shape=(n_pair,n_truth))
        predicted=csr_matrix((np.ones(n_pair),(np.arange(n_pair),col)),shape=(n_pair,n_cluster))
        seeds.append(dict(cells=cells,truth=truth,predicted=predicted,row=row,col=col))
    return dict(kind='sparse_observed_contingency',n_groups=n_groups,seeds=seeds)
def score(tables,weights):
    values=[]
    for seed in tables['seeds']:
        counts=np.asarray(weights@seed['cells']);rows=np.asarray(counts@seed['truth']);cols=np.asarray(counts@seed['predicted']);total=counts.sum(-1,keepdims=True)
        probability=counts/np.maximum(total,1);py=rows/np.maximum(total,1);pc=cols/np.maximum(total,1)
        denominator=py[:,seed['row']]*pc[:,seed['col']]
        mi=np.sum(np.where(probability>0,probability*np.log(np.maximum(probability,1e-300)/np.maximum(denominator,1e-300)),0),axis=-1)
        hy=-np.sum(np.where(py>0,py*np.log(np.maximum(py,1e-300)),0),axis=-1);hc=-np.sum(np.where(pc>0,pc*np.log(np.maximum(pc,1e-300)),0),axis=-1)
        values.append(np.divide(2*mi,hy+hc,out=np.ones_like(mi),where=(hy+hc)>1e-15))
    return np.stack(values,axis=-1)
def group_count(tables):return tables['n_groups']if isinstance(tables,dict)else len(tables)
