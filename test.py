
import example

PARAM_DATA_N = 17770
PARAM_DATA_Q = 999
PARAM_DATA_D = 300
TOP_K = 10
TOP_B = 300
D_UP = 1024
save_files = True
S = 5
ceos = example.OneCEOs(PARAM_DATA_N,PARAM_DATA_Q,
PARAM_DATA_D,TOP_K,TOP_B,save_files,D_UP)


'''
ceos = example.TwoCEOs(PARAM_DATA_N,PARAM_DATA_Q,
PARAM_DATA_D,TOP_K,TOP_B,save_files,UP)
example.sCEOsEst(PARAM_DATA_N,PARAM_DATA_Q,
PARAM_DATA_D,TOP_K,TOP_B,save_files,D_UP,S)
'''
X_path = "./_X_17770_300.txt"
Q_path = "./_Q_999_300.txt"

print("begin read!!!")
ceos.read_matrix_x(X_path)
ceos.read_matrix_q(Q_path)

ceos.build_Index()
ceos.find_TopK()



