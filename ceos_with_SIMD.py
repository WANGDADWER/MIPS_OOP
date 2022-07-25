import time

import example
import numpy as np
from utils import fvecs_read



'''
base_path = '/home/wang/Desktop/data/cifar60k/cifar60k_base.fvecs'
query_path = '/home/wang/Desktop/data/cifar60k/cifar60k_query.fvecs'
xb = fvecs_read(base_path).T #[N,D]
xq = fvecs_read(query_path).T
'''
base_path = '/home/wang/Desktop/data/gist/gist_base.fvecs'
query_path = '/home/wang/Desktop/data/gist/gist_query.fvecs'
xb = fvecs_read(base_path).T #[N,D]
xq = fvecs_read(query_path).T


(D, N) = xb.shape
(_, Q) = xq.shape

K = 10
TOP_B = 300
D_UP = 1024
save_files = False

ceos = example.TwoCEOs(N,Q,D,K,TOP_B,save_files,D_UP)

ceos.read_X_from_np(xb)  #numpy [D, N]
ceos.read_Q_from_np(xq)  #numpy [D, Q]

ceos.build_Index()
st = time.time()
top_K = ceos.find_TopK() #numpy [K, Q]
print("query time : {}".format(round(time.time() - st,3)))


topK_Acc = np.zeros((Q, 1))
for q in range(Q):

    # Get BF topK
    exactDOT = np.matmul(xb.T, xq.T[q, :].transpose())  # Exact dot products
    topK = np.argsort(-exactDOT)[:K]  # topK MIPS indexes
    topK_Acc[q] = len(np.intersect1d(top_K, topK)) / K  # Get topK MIPS accuracy

print("Accuracy of Top", K, "MIPS with OneCEOs", "concomitants and ", TOP_B, "dot products in post-processing: ",
          np.mean(topK_Acc))
