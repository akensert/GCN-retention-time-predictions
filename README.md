## Graph Convolutional Networks (GCNs) for predicting retention times
<br>

### Requirements
1. Python (version ~= 3.6)
2. Pip (package manager) (version ~= 20.0.2)
3. Conda (version ~= 4.9.1)

### Install
1. Create rdkit-env via conda:<br>
`conda create -c conda-forge -n my-rdkit-env rdkit=2020.09.01`
2. Activate 'my-rdkit-env' and install packages:<br>
`conda activate my-rdkit-env`<br>
`conda install tensorflow-gpu==2.4.1`<br>
`conda install tqdm==4.59.0`<br>
`conda install scikit-learn==0.23.2`<br>
`conda install jupyter`<br>
`conda install matplotlib` [Optional]<br>
Try `conda install -c conda-forge [package]` instead of `conda install [package]` if the latter doesn't work.

### 1. Create data sets
Navigate into `src/` and run the following from the terminal to create tf-records from all the .csv files in `input/datasets/`: `python create_tf_records.py --num_threads=8`. This may take up to 30 minutes to run.

### 2. Training, validating and testing
Navigate into `notebooks/` and run from terminal: `jupyter notebook`. Open and run `evaluation.ipynb`. This may take weeks to run. To reduce run time, reduce the number of searches (`NUM_SEARCHES`) to be performed.

### 3. Compute saliencies
Navigate into `notebooks/` and run from terminal: `jupyter notebook`. Open and run `saliency.ipynb`. This make take up to an hour to run.
