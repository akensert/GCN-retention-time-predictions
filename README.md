## Graph nets for predicting physicochemical properties

### Requirements
1. Python (version ~= 3.6)
2. Pip (package manager) (version ~= 20.0.2)
3. Conda (version ~= 4.9.1)

### Install
1. Create rdkit-env via conda:<br>
`conda create -c conda-forge -n my-rdkit-env rdkit=2020.09.01`
2. Install packages:<br>
`conda activate my-rdkit-env`<br>
`conda install tensorflow-gpu==2.4.1`<br>
`conda install tqdm==4.59.0`<br>
`conda install scikit-learn==0.23.2`<br>
`conda install jupyter`<br>
`conda install matplotlib`<br>
Try `conda install -c conda-forge [package]` instead of `conda install [package]` if the latter don't work.

### Create data sets
Navigate into `src/` and run the following from the terminal to create tf-records from all the .csv files in `input/datasets/`:<br>
`python create_tf_records.py –dataset_path=../input/datasets/* --num_threads=4`<br>
This may take up to 30 minutes depending on the number of threads used.

### Training

### Testing

### Saliency
