
NUM_SEARCHES = 20   # number of random hyper-parameter searches
NUM_REPLICATES = 1  # number of replications of the best performing model

# random seed for train/valid/test splits
seeds = {
    "splits": 42
}

# datasets used, including their train/valid/test sizes and splitting mode
datasets = {
 "ESOL": {
   "train_frac": 0.8,
   "valid_frac": 0.1,
   "test_frac": 0.1,
   "mode": "random"
  },
 "SMRT": {
    "train_frac": 0.8,
    "valid_frac": 0.1,
    "test_frac": 0.1,
    "mode": "random"
    },
 "FreeSolv": {
    "train_frac": 0.8,
    "valid_frac": 0.1,
    "test_frac": 0.1,
    "mode": "random"
    },
 "Lipophilicity": {
    "train_frac": 0.8,
    "valid_frac": 0.1,
    "test_frac": 0.1,
    "mode": "random"
    },
 "Fiehn_HILIC": {
    "train_frac": None,
    "valid_frac": None,
    "test_frac": None,
    "mode": "index"
    },
 "RIKEN": {
    "train_frac": None,
    "valid_frac": None,
    "test_frac": None,
    "mode": "index"
    },

  "Beck": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Cao_HILIC": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Eawag_XBridgeC18": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "FEM_lipids": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "FEM_long": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "FEM_orbitrap_plasma": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "FEM_orbitrap_urine": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "FEM_short": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "IPB_Halle": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "kohlbacher": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Krauss": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Krauss_21": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "LIFE_new": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "LIFE_old": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Matsuura": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "MTBLS20": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "MTBLS36": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "MTBLS38": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "MTBLS87": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Nikiforos": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Stravs": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Stravs_22": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Taguchi_12": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Takahashi": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Tohge": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "Toshimitsu": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "UFZ_Phenomenex": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
  "UniToyama_Atlantis": {
    "train_frac": 0.7,
    "valid_frac": 0.1,
    "test_frac": 0.2,
    "mode": "random"
    },
}
