# fish-classifier-tpot
Fish species classifier using TPOT.

# How to Set up and Run Demo
```
$ conda create -n fish-classifier-tpot python=3.6 pip
$ conda activate fish-classifier-tpot
$ conda install numpy scipy scikit-learn pandas joblib
$ pip install deap update_checker tqdm stopit
$ pip install xgboost
$ pip install "dask[delayed]" "dask[dataframe]" dask-ml fsspec>=0.3.3
$ pip install scikit-mdr skrebate
$ pip install tpot
$ python fish.py
```
