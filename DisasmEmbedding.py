import sys
sys.path += ["DataIngest/Asm2Vec"]

import argparse
import os
import time
import torch
from torch import nn
from tqdm import tqdm
from typing import List, Tuple

from DataIngest.Asm2Vec.Asm2VecModel import ASM2VEC
from DataIngest.Asm2Vec.DataType import Function, Tokens



# --- Disassembly Embedding Class --------------------------------------------------------------------------------------
class DisasmEmbedding(nn.Module):
    '''
    
    '''
    def __init__(self, input_dir: str, save_path: str) -> None:
        '''
        
        '''
        super(DisasmEmbedding, self).__init__()
        self.input_dir: str    = input_dir
        self.save_path: str    = save_path
        self.functions: List   = []
        self.tokens:    Tokens = Tokens()

        self.tensors: Tuple[torch.Tensor, torch.Tensor] = None

        self.device:           str   = "cpu"
        self.epochs:           int   = 3
        self.embedding_size:   int   = 64
        self.batch_size:       int   = 128
        self.learning_rate:    float = 0.02
        self.negative_samples: int   = 25

        self.model = None


    def load_data(self) -> None:
        '''
        
        '''
        print("[INFO] Indexing Data")
        # Index files for training
        file_list = []
        for inode in os.listdir(self.input_dir):
            subdir = os.path.join(self.input_dir, inode)
            for file in os.listdir(subdir):
                file_path = os.path.join(subdir, file)
                for asm_file in os.listdir(file_path):
                    file_list.append(os.path.join(file_path, asm_file))
        # Load functions and tokens
        self.functions = []
        self.tokens    = Tokens()
        for file in tqdm(file_list):
            with open(file, 'r') as f_in:
                fn = Function.load(f_in.read())
                self.functions.append(fn)
                self.tokens.add(fn.tokens())
        self.preprocess()
    

    def preprocess(self) -> None:
        '''
        
        '''
        x, y = [], []
        for i, fn in enumerate(self.functions):
            for seq in fn.random_walk():
                for j in range(1, len(seq) - 1):
                    x.append([i] + [self.tokens[token].index for token in seq[j-1].tokens() + seq[j+1].tokens()])
                    y.append([self.tokens[token].index for token in seq[j].tokens()])
        self.tensors = (torch.tensor(x), torch.tensor(y))
    

    def train(self, do_accuracy: bool = True, callback=None) -> None:
        '''
        
        '''
        # Initialize model
        self.model = ASM2VEC(self.tokens.size(), function_size=len(self.functions), embedding_size=self.embedding_size).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Load data
        loader = torch.utils.data.DataLoader(AsmDataset(*self.tensors), batch_size=self.batch_size, shuffle=True)
        # Train model
        self.model.train()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}")
            start = time.time()
            loss_sum, loss_count, accs = 0.0, 0, []

            for i, (inp, pos) in enumerate(loader):
                neg = self.tokens.sample(inp.shape[0], self.negative_samples)
                loss = self.model(inp.to(self.device), pos.to(self.device), neg.to(self.device))
                loss_sum, loss_count = loss_sum + loss, loss_count + 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i == 0 and do_accuracy:
                    probs = self.model.predict(inp.to(self.device), pos.to(self.device))
                    accs.append(accuracy(pos, probs))

            if callback:
                callback({
                    'model': self.model,
                    'tokens': self.tokens,
                    'epoch': epoch,
                    'time': time.time() - start,
                    'loss': loss_sum / loss_count,
                    #'accuracy': torch.tensor(accs).mean() if calc_acc else None
                })
            print(f"Loss: {loss} || Accuracy: {torch.tensor(accs).mean()}")


    def save_model(self) -> None:
        try:
            torch.save({
                'model_params': (
                    self.model.embeddings.num_embeddings,
                    self.model.embeddings_f.num_embeddings,
                    self.model.embeddings.embedding_dim
                ),
                'model': self.model.state_dict(),
                'tokens': self.tokens.state_dict(),
            }, self.save_path)
            print(f"[INFO] Saved embedding model to {self.save_path}")
        except Exception as e:
            print(f"[ERR]Unable to save embedding model\n{e}")

    
    def load_model(self) -> None:
        '''
        
        '''
        # Initialize pointer to checkpoint file
        checkpoint = torch.load(self.save_path, map_location=self.device)
        # Load tokens
        self.tokens = Tokens()
        self.tokens.load_state_dict(checkpoint["tokens"])
        # Load model
        self.model = ASM2VEC(*checkpoint["model_params"])
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(self.device)
    

    def forward(self, input) -> torch.Tensor:
        '''
        
        '''
        try:
            ret_tensor = self.model.v(input)
            return ret_tensor
        except Exception as e:
            print(f"[ERR] Unable to vectorize the given input\n{e}")



def accuracy(y, probs):
    return torch.mean(torch.tensor([torch.sum(probs[i][yi]) for i, yi in enumerate(y)]))



# --- Assembly Dataset Class -------------------------------------------------------------------------------------------
class AsmDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]


# --- Script Code ------------------------------------------------------------------------------------------------------
def parseArgv() -> argparse.Namespace:
    '''
    
    '''
    parser = argparse.ArgumentParser(prog="Assembly embedding layer training script",
                                     description="Train the embedding layer of the assembly encoder.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mode",
                        help="Mode of operation for embedding to operate under.",
                        type=str,
                        choices=["train", "embed"],
                        required=True)
    parser.add_argument("--assembly",
                        help="Directory containing preprocessed assembly files.",
                        type=str,
                        required=True)
    parser.add_argument("--model",
                        help="Path associated with model.",
                        type=str,
                        required=True)
    return parser.parse_args()


# def main(argc, argv):
if __name__  == "__main__":
    argv = parseArgv()
    if(argv.mode == "train"):
        disasmEmbed = DisasmEmbedding(argv.assembly, argv.model)
        disasmEmbed.load_data()
        disasmEmbed.train()
        disasmEmbed.save_model()
    elif(argv.mode == "embed"):
        disasmEmbed = DisasmEmbedding(argv.assembly, argv.model)
        disasmEmbed.load_model()
        tens = torch.randint(1, 64, (128, 7))
        vec = disasmEmbed(tens)
        print(vec)
