import requests
import zipfile
import os
import sys
import string
from tok import CharacterTokenizer
import torch
from torch import Tensor
import random
from jaxtyping import Int
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from tqdm.auto import tqdm
from bfn import get_gpt_net, BayesianFlowNetwork, get_param_groups
import matplotlib.pyplot as plt
import wandb
from dataclasses import dataclass, asdict
import logging
import math
import yaml
from logging import getLogger

TEXT8_URL = "https://mattmahoney.net/dc/text8.zip"
logging.basicConfig(level=logging.INFO)
LOGGER = getLogger(__name__)

def download_text8(out_path: str) -> None:
    """
    Downloads the text8 dataset and saves it to the specified path.
    """
    LOGGER.info(f"Downloading text8 dataset to {out_path}")
    r = requests.get(TEXT8_URL)
    with open(out_path, "wb") as f:
        f.write(r.content)

def extract_text8(zip_path: str, out_path: str) -> None:
    """
    Extracts the text8 dataset from the specified zip file to the specified path.
    """
    LOGGER.info(f"Extracting text8 dataset from {zip_path} to {out_path}")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(out_path)

def get_text8(out_path: str) -> str:
    """
    Downloads and extracts the text8 dataset to the specified path.
    The returned path is the path to the extracted text8 file.
    """
    zip_path = out_path + ".zip"
    if not os.path.exists(zip_path):
        download_text8(zip_path)
    if not os.path.exists(out_path):
        extract_text8(zip_path, out_path)
    return os.path.join(out_path, "text8")

def chunk_dataset(tokens: Int[Tensor, "1 D"], chunk_size: int, rng: random.Random) -> Int[Tensor, "n_chunks D"]:
    """
    Chunks the dataset into sequences of length chunk_size, randomly shuffled.
    """
    chunked = list(tokens.chunk(tokens.shape[-1] // chunk_size, dim=-1))
    rng.shuffle(chunked)
    if chunked[-1].shape != chunked[0].shape:
        # if the last chunk is smaller than the rest, drop it
        chunked = chunked[:-1]
    return torch.cat(chunked, dim=0)

class Text8Dataset(Dataset):

    def __init__(self, tokens: Int[Tensor, "1 D"], chunk_size: int, rng: random.Random):
        super().__init__()
        self.tokens = tokens
        self.chunk_size = chunk_size
        self.rng = rng
        self.tokens_chunked = chunk_dataset(self.tokens, chunk_size, rng)

    def __len__(self) -> int:
        return self.tokens.shape[-1] // self.chunk_size

    def __getitem__(self, index: int) -> Int[Tensor, "1 D"]:
        return self.tokens_chunked[index]

@dataclass
class TrainConfig:
    vocab_size: int
    lr: float = 1e-4
    weight_decay: float = 1e-1
    batch_size: int = 64
    D: int = 256
    seq_len: int = 256
    hdim: int = 768
    num_layers: int = 24
    num_heads: int = 12
    dropout: float = 0.0
    seed: int = 0
    adam_betas: Tuple[float, float] = (0.9, 0.98)
    n_batches: int = 1_200_000
    net_save_path: str = "net.pt"
    eval_every: int = 100
    eval_batches: int = 25

    def __post_init__(self):
        assert self.D == self.seq_len, "D must be equal to seq_len"

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

# TODO: read config from file / command line
# TODO: better shuffling - it should take different random slices each epoch
# TODO: toy dataset that just cycles through the alphabet, to test if it's learning anything
# TODO: try adding causal mask to gpt

if __name__ == "__main__":
    # download text8
    os.makedirs("data", exist_ok=True)
    path = get_text8("data/text8")
    LOGGER.info(f"Using text8 dataset at {path}")
    with open(path, "r") as f:
        text = f.read()
    
    # construct tokenizer
    vocab = string.ascii_lowercase + " "
    tok = CharacterTokenizer(vocab, model_max_length=256)

    # verify that encode+decode == identity
    a = tok.decode(tok.encode(text[:100]))
    assert a == text[:100], f"encode+decode != identity: {a} != {text[:100]}"

    # encode text
    LOGGER.info("Encoding text")
    tokens = tok.encode(text, return_tensors="pt")

    # construct train config
    if len(sys.argv) > 1:
        _config = load_config(sys.argv[1])
        _config['vocab_size'] = tok.vocab_size
        CONFIG = TrainConfig(**_config)
    else:
        CONFIG = TrainConfig(vocab_size=tok.vocab_size)

    # construct dataset
    rng = random.Random(CONFIG.seed)
    torch_rng = torch.Generator()
    torch_rng.manual_seed(CONFIG.seed)
    dataset = Text8Dataset(tokens, CONFIG.seq_len, rng)
    train_size = int(len(dataset) * 0.9)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG.batch_size,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
        generator=torch_rng,
    )
    dataloader_iter = iter(dataloader)
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=CONFIG.batch_size,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
        generator=torch_rng,
    )
    eval_dataloader_iter = iter(eval_dataloader)
    LOGGER.info(f"Dataset has {len(eval_dataset)} chunks of size {CONFIG.seq_len}.")

    # construct model
    LOGGER.info("loading model...")
    gpt = get_gpt_net(D=CONFIG.D + 1, vocab_size=CONFIG.vocab_size, hidden_dim=CONFIG.hdim, num_layers=CONFIG.num_layers, num_heads=CONFIG.num_heads, dropout=CONFIG.dropout) # D+1 because timestep is added to the last position in the sequence
    LOGGER.info('gpt loaded')
    bfn = BayesianFlowNetwork(gpt, D=CONFIG.D, vocab_size=CONFIG.vocab_size, beta=3.0)
    LOGGER.info('bfn loaded')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bfn = bfn.to(device)

    # get param groups / optimizer
    param_groups = get_param_groups(bfn, weight_decay=CONFIG.weight_decay)
    optim = AdamW(param_groups, lr=CONFIG.lr, betas=CONFIG.adam_betas)

    # init wandb
    print(asdict(CONFIG))
    wandb.init(project="bfn", entity="conjecture_engineering", config=asdict(CONFIG))

    # train
    losses = []
    pbar = tqdm(range(CONFIG.n_batches))
    for i in pbar:
        optim.zero_grad()

        try:
            X = next(dataloader_iter).to(device)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            X = next(dataloader_iter).to(device)
        loss = bfn.process(X)
        loss.backward()

        optim.step()
        wandb.log({"train/loss": loss.item(), "train/ppl": 2 ** loss.item(), "train/bpc": loss.item() / math.log(2)})
        losses.append(loss.item())
        if i % 1000 == 0:
            pbar.set_description(f"loss: {loss.item():.4f}")
        if i % CONFIG.eval_every == 0:
            bfn.eval()
            with torch.no_grad():
                eval_losses = []
                for _ in range(CONFIG.eval_batches):
                    try:
                        X = next(eval_dataloader_iter).to(device)
                    except StopIteration:
                        eval_dataloader_iter = iter(eval_dataloader)
                        X = next(eval_dataloader_iter).to(device)
                    loss = bfn.process(X)
                    eval_losses.append(loss.item())
                avg_loss = sum(eval_losses) / len(eval_losses)
                avg_ppl = 2 ** avg_loss
                wandb.log({"eval/loss": sum(eval_losses) / len(eval_losses), "eval/ppl": avg_ppl, "eval/bpc": avg_loss / math.log(2)})
                LOGGER.info(f"eval loss: {avg_loss:.4f}, eval ppl: {avg_ppl:.4f}")
            bfn.train()

    
    # save model at end of training
    LOGGER.info(f"Saving model to {CONFIG.net_save_path}")
    torch.save(bfn.net.state_dict(), CONFIG.net_save_path)




