# Hyperbolic Embedding using HTorch
This module contains an example `embedding.ipynb` to train embedding using HTorch based on the methodology proposed in [Poincaré Embeddings for Learning Hierarchical Representations](https://github.com/facebookresearch/poincare-embeddings). We conduct experiments on the [WordNet](https://wordnet.princeton.edu/) Mammals dataset with 1181 nodes and 6541 edges, custom dataset can also be used. Now the model of the hyperbolic embedding supports `PoincareBall`, `Lorentz` and `HalfSpace`.


## Install
### Requirements
- pytorch
- nltk
- scikit-learn
- pandas
- h5py
- cython
- tqdm
- numpy
or alternatively, do `pip install -r requirements.txt`.

### Procedures
1. run following commands:
```python
python setup.py build_ext --inplace 
```
2. Generate the transitive closure of the WordNet Mammals dataset:
```python
cd wordnet
python transitive_closure.py
```
3. Play with the `embedding.ipynb` notebook! 

Both model weight are initalized with float64.

### Meta-parameters and hyperparameters
We recommend training with float64 to avoid potential imprecision problems, which cause `inf` and `NaN`. To modify the hyperparameters or the datatype for each model, edit the cell below _Hyperparameter_ section in the notebook.