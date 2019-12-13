import argparse

#from models import *
#from dataloader import *

# Set Seed
'''
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

train_dir="./data/train"
    test_dir = "./data/test"
    doc_dir="./data/doc"
    get_dataloader(train_dir, test_dir, doc_dir, split="All")
'''




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Setting parser of dataset
    parser.add_argument("--train_dir", type=str, required=True, help='The directory of train data.')
    parser.add_argument("--test_dir", type=str, required=True, help='The directory of test data.')
    parser.add_argument("--doc_dir", type=str, required=True, help='The directory of document data.')

    