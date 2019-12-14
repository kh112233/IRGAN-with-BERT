import argparse

from models import *
from dataloader import *
from trainer import IRGANTrainer

# Set Seed

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def main(args):
    train_dir = args.train_dir
    test_dir = args.test_dir
    doc_dir = args.doc_dir
    dataloder = get_dataloader(train_dir, test_dir, doc_dir, batch_size = 128, split="All")

    G = Generator() 
    D = Generator()

    trainer = IRGANTrainer(G, D, train_dir, test_dir, doc_dir)
    trainer.G_get_top_k(dataloder)






if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # Setting parser of dataset
    parser.add_argument("--train_dir", type=str, required=True, help='The directory of train data.')
    parser.add_argument("--test_dir", type=str, required=True, help='The directory of test data.')
    parser.add_argument("--doc_dir", type=str, required=True, help='The directory of document data.')
    
    main(parser.parse_args())

    # python main.py --train_dir ./data/train --test_dir ./data/test --doc_dir ./data/doc