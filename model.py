import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

"""
parser = argparse.ArgumentParser(description='this is a demonstration script')
# here we add an argument to the parser, specifying the expected type, a help message, etc
parser.add_argument('-batch_size', type=str, required=True, help='please provide a batch size')

args = parser.parse_args()

# now we can use the argument value in our program
print(f'batch_size: {args.batch_size}')
"""

# important to node cuda is not M1 compatible, instead we use mps
# sets the device to MPS (Metal Performance Shaders) for Apple Silicon Macs, or CPU otherwise
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

# hyperparameters
batch_size = 64 #args.batch_size       # number of sequences processed in parallel
block_size = 128       # context length for each input
max_iters = 200     # number of training iterations
learning_rate = 3e-4 # learning rate for optimizer
eval_iters = 100    # number of steps for loss estimation
n_embd = 384         # embedding size
n_head = 1           # number of attention heads
n_layer = 1         # number of transformer blocks
dropout = 0.2        # dropout rate


# loads the training text into a string
chars = ""
with open('vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)


# character level tokenizer
string_to_int = { ch:i for i,ch in enumerate(chars)}
int_to_string = { i:ch for i,ch in enumerate(chars)}

# encode: turns a string into a list of integers
# decode: turns a list of integers back into a string
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])


class Head(nn.Module):
    # one head of self-attention
    
    def __init__(self, head_size):
        # calls the constructor of nn.Module
        super().__init__()
        
        # each token is transformed into a query, key, and value using learned linear layers
        # head_size is the dimension of this head's internal representation
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # creates a lower-triangular mask (tril) to prevent tokens from attending to future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        # applies dropout to the attention weights to regularize training
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        
        # input x: shape (Batch, Time, Channels)
        # outputs k and q: shape (B, T, head_size)
        B,T,C = x.shape
        k = self.key(x) # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        
        # compute raw attention scores via scaled dot product.
        # shape becomes (B, T, T) representing attention between all pairs of tokens
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        # masks out future tokens (above the diagonal), setting them to -inf so softmax zeroes them
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        
        # converts attention scores to probabilities and applies dropout
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # uses the attention weights to compute a weighted sum of values
        v = self.value(x) # # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    # multiple heads of attention in parallel
    
    def __init__(self, num_heads, head_size):
        # combines multiple Heads in parallel
        # the outputs are concatenated and passed through a final linear layer to project back to n_embd
        
        # calls the constructor of nn.Module
        super().__init__()
        
        # creates multiple Head modules, each representing a self-attention head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # pass heads through a final linear projection to fuse their information and bring dimensionality back
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        
        # apply dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # concatenates each head’s output, projects it back to the full embedding size
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # apply dropout to prevent overfitting
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    # a simple linear layer followed by a non-linearity
    
    def __init__(self, n_embd):
        # calls the constructor of nn.Module
        super().__init__()
        
        # this part handles the "thinking" after the "communication" of attention
        # nn.Linear -> expands the embedding size 4x, allows richer transformations
        # nn.ReLU() -> adds non-linearity, helping model complex patterns
        # nn.Linear -> shrinks it back to original size
        # nn.Dropout -> helps prevent overfitting
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # trandformer block: communication followed  by computation
    
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        
        # calls the constructor of nn.Module
        super().__init__()
        
        head_size = n_embd // n_head
        
        # lets tokens attend to each other (context building)
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # lets each token process its information
        self.ffwd = FeedForward(n_embd)
        
        # normalizes inputs to stabilize training
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        y = self.sa(x)
        # residual connections (x + y) –> preserves input and adds change instead of replacing it
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        # calls the constructor of nn.Module
        super().__init__()
        
        # creates an embedding table, converts token indices into vectors
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # creates a positional embedding table, adds info about token positions
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # how many decoder blocks we have running sequentially
        # stack of transformer layers, each consisting of attention + feedforward
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # make n_layer blocks
        
        # add to the end of our network, final normalization
        self.ln_f = nn.LayerNorm(n_embd)
        
        # outputs raw logits over vocabulary
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # this line tells PyTorch: “go through all modules and apply _init_weights to each one”
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        # if the module is a fully connected layer (nn.Linear), we initialize it using normal distribution
        if isinstance(module, nn.Linear):
            # initializes the weights with random numbers drawn from a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            # biases are initialized to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        # same initialization for embeddings so that each token and position starts with small random vectors
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, index, targets=None):      
        # idx and targets are both (B,T) tensor of integers
        B, T = index.shape
        
        # token embeddings
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        
        # position embeddings for each time step
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        
        # combined positional input
        x = tok_emb + pos_emb # (B,T,C)
        
        # run through stacked transformer blocks
        x = self.blocks(x) # (B,T,C)
        
        # normalize
        x = self.ln_f(x) # (B,T,C)
        
        # final vocab scores -> shape (B, T, vocab_size)
        logits = self.lm_head(x)
        
        
        # flatten (B, T, vocab_size) → (B*T, vocab_size)
        # compute cross-entropy loss between logits and target tokens
        if targets is None: # inference mode
            loss = None
        else:
            # batch, time, channels(vocabulary)
            # B (Batch Size) -> Number of sequences processed at once
            # T (Time Steps / Sequence Length) -> Number of tokens in each sequence
            # C (Vocabulary Size / Channels) -> Number of possible tokens
            B, T, C = logits.shape
            
            # reshape batch and time into a single dimension 
            # so that each token is treated as a separate training example
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # targets also reshaped into a single B*T vector
            
            # compute cross-entropy loss to measure how far our predictions (logits) are from the true targets
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop index to thel;ast block_size tokens
            index_cond = index[:, -block_size:]

            # get the predictions
            logits, loss = self.forward(index_cond)
            
            # extracts only the last time step’s logits
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply softmax to get probabilities for each possible next token
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # samples one token index from the probability distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

    
def load_model():
    """Load the trained model"""
    model = GPTLanguageModel(vocab_size)
    print('loading model parameters')
    with open('model-01.pkl', 'rb') as f:
        model = pickle.load(f)
    print('loaded successfully')
    return model.to(device)


# For command line usage (if this file is run directly)
if __name__ == "__main__":
    m = load_model()
    
    while True:
        prompt = input("Prompt:\n")
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
        generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
        print(f'Completion:\n{generated_chars}')