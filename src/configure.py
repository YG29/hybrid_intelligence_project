DATA_ROOT = "./data"
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
OUTPUT_CSV = "first_round_predictions.csv"
RANDOM_SEED = 192
MODEL_PATH = "resnet18_gtsrb_finetuned.pth"
PREDICTIONS_CSV = "newdataset.csv"
ANNOTATIONS_CSV = "human_annotations.csv"

DEL_NUM_EPOCHS = 30
DEL_LEARNING_RATE = 1e-4
DELEGATION_PATIENCE = 5

DEVICE = "cpu"

TEST_PREDICTIONS_CSV = "test_predictions.csv"
TEST_HUMAN_CSV = "test_delegation_human.csv"
TEST_MODEL_CSV = "test_delegation_model.csv"
