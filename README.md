## Graph Convolutional Networks for predicting physicochemical properties
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

### Create data sets
Navigate into `src/` and run the following from the terminal to create tf-records from all the .csv files in `input/datasets/`:<br>
`python create_tf_records.py –dataset_path=../input/datasets/* --num_threads=4`<br>
This may take up to 30 minutes depending on the number of threads used.

### Training and testing
Navigate into `notebooks/` and run from terminal: `jupyter notebook`. Open and run `01_evaluate-GCNs.ipynb`. In brief, this notebook runs a 20x random search for both the GCN and RGCN.

### Saliency
Navigate into `notebooks/` and run from terminal: `jupyter notebook`. Open and run `05_saliency.ipynb`. In brief, this notebook will, for Fiehn HILIC and RIKEN: (1) train a GCN on the data set, (2) compute saliencies (for each data point) based on the trained GCN, (3) draw saliencies on corresponding 2-D representations of the molecules, and (4) save the drawings to `../output/saliency/*`.
