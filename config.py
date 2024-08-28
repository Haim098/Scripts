import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# File names
FEATURES_FILE = 'elliptic_txs_features.csv'
CLASSES_FILE = 'elliptic_txs_classes.csv'
EDGELIST_FILE = 'elliptic_txs_edgelist.csv'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature engineering
TIME_STEP_RANGE = {
    'train': (1, 30),
    'valid': (31, 40),
    'test': (41, 49)
}

# Anomaly detection
ANOMALY_THRESHOLD = 0.5  # This is a placeholder. Adjust based on your needs.

# Visualization
TSNE_PERPLEXITY = 30
TSNE_N_COMPONENTS = 2