# MIPS_OOP

requirement: Eigen 3.4, Boost, OpenMP
~~~bash
sudo apt install libboost-container-dev libeigen3-dev
mkdir  build
cd build
cmake ..
make
~~~




## Update
Multithreading is enabled
and the use of inner products in the Hnsw library.
(compile option -O3 -march-native)


an example in test.py

data source:
https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html

data format : fvecs
