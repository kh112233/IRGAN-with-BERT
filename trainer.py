import torch
from torch import nn
import torch.optim as optim

import numpy as np

from models import *

class IRGANTrainer:

    def __init__(self, G:Generator, D:Discriminator,
                 train_dir:str, test_dir:str, doc_dir:str,
                 batch_size = 2, accm_batch_size = 32, G_lr = 2e-5, D_lr = 2e-5,
                 with_cuda: bool = True):
        """
        :param G: BERT Generator model
        :param D: Bert Discirminator model
        :param train_dir: Path directory of train data
        :param test_dir: Path directory of test data
        :param doc_dir: Path directory of document data
        :param batch_size: Training batch size
        :param accm_batch_size: Accumulate gradient batch size
        :param G_lr: Genrator model learning rate of AdamW
        :param D_lr: Discriminator model learning rate of AdamW
        :param with_cuda: training with cuda
        """
        
        # Setup cuda device for BERT model training
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # Setting generator and discriminator bert model
        self.G = G
        self.D = D 

        # Setting Directory path of train dataset, test dataset, document data
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.doc_dir = doc_dir

        # Setting hyper-parameter of training
        self.batch_size = batch_size
        assert accm_batch_size % self.batch_size == 0
        self.accm_steps = accm_batch_size // self.batch_size

        # Setting the AdamW optimizer with hyper-parameter
        self.G_optimizer = optim.AdamW(self.G.parameters(), lr=G_lr) # suggest lr: 2e-5~3e-5
        self.D_optimizer = optim.AdamW(self.D.parameters(), lr=D_lr) # suggest lr: 2e-5~3e-5

    def G_get_top_k(self, dataloader, k=100):
        self.G.eval()
        with torch.no_grad():
            sigmoid = nn.Sigmoid()
            y_scores = {}

            for step, batch in enumerate(dataloader):
                print(f"{step+1} / {len(dataloader)} | {(step+1)/len(dataloader)*100:.2f}%    \r",end="")

                batch = [*(tensor for tensor in batch)]
                ids, token_type, attention_mask, q_name, d_name = batch
                ids.cuda()
                token_type.cuda()
                attention_mask.cuda()

                logits = self.G(input_ids=ids, token_type_ids=token_type, attention_mask=attention_mask)
                scores = sigmoid(logits[0])
                #y_scores = np.concatenate([y_scores, scores.cpu().numpy().squeeze(-1)])
                scores = scores.cpu().numpy().squeeze(-1)
                for q_name, d_name, score in zip(q_name, d_name, scores):
                    if q_name in y_scores:
                        values = y_scores[q_name]
                        values.append(score)
                        y_scores[q_name] = values
                    else:
                        y_scores[q_name] = [score]
            print(y_scores)

        return loss, accuracy

