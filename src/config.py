DATA_PATH = "data/raw/acct_transaction.csv"
LABEL_PATH = "data/raw/acct_alert.csv"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

MU = 10
LAMBDA = 20
GENERATION = 20

FITNESS_WEIGHT = {
    "f1": 0.7,
    "recall": 0.3,
    "complexity": 0.01
}