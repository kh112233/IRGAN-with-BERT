from torch import nn
import torch.optim as optim

from models import *

class IRGANTrainer:

    def __init__(self, bert_finetune:BERTFinetuneModel,
                 train_dataloader:DataLoader, val_dataloader=None, test_dataloader=None, test_dict=None,
                 batch_size = 2, accm_batch_size = 32, lr = 2e-5,
                 with_cuda: bool = True):
        """
        :param bert_finetune: BERT finetune model
        :param train_dataloader: train dataset data loader
        :param val_dataloader: valid dataset data loader (can be None)
        :param test_dataloader: test dataset data loader (can be None)
        :param batch_size: batch size of data loader 
        :param accm_batch_size: accumulate gradient batch size
        :param lr: learning rate of AdamW
        :param with_cuda: training with cuda
        """
