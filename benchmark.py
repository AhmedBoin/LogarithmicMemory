import time
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

from Source.memory import LogarithmicMemory

        
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads # wordEmbed 128 // numOfHeads 8 = 16
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim) # linear layer (W_O) to generate output
        
    def forward(self, x): # x = [batch_size, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.size()
        K = self.key(x) # [batch_size, seq_len, embed_dim]
        Q = self.query(x) # [batch_size, seq_len, embed_dim]
        V = self.value(x) # [batch_size, seq_len, embed_dim]
        
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, numOfHeads, seq_len, head_dim] 
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, numOfHeads, seq_len, head_dim]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # [batch_size, numOfHeads, seq_len, head_dim]
        
        
        attention_weight = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim **0.5), dim=-1) #Q @ K.transpose(-1, -2)
        output = torch.matmul(attention_weight, V)  #output = attention_weight @ V
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(output)
    
def masked_softmax(x): # x = [batch_size, seq_len, seq_len]
    upper_triangle_mask = torch.triu(torch.ones_like(x), diagonal=1).bool()
    x[upper_triangle_mask] = float('-inf')
    return F.softmax(x, dim=-1)


if __name__ == "__main__":
    with torch.no_grad():

        # Parameters
        batch_size = 1
        embed_dim = 32
        heads = 1
        memory_banks = 1

        # Initialize modules
        memory = LogarithmicMemory(embed_dim, heads, memory_banks)
        attention = MultiHeadAttention(embed_dim, heads)

        # Data collection
        parallel_seq_lengths = [2048 * (2 ** n) for n in range(5)]
        parallel_memory_times = []
        parallel_attention_times = []
        sequential_seq_lengths = [128 * (2 ** n) for n in range(5)]
        sequential_memory_times = []
        sequential_attention_times = []

        # Parallel Time
        memory.parallelize = True
        memory = memory#.to("mps")
        attention = attention#.to("mps")
        print("Parallel:")
        for seq_len in parallel_seq_lengths:
            print("seq_lengths:", seq_len)
            x = torch.randn(batch_size, seq_len, embed_dim)#.to("mps")

            # Memory
            start = time.time()
            memory(x)
            parallel_memory_times.append(time.time() - start)
            print("Memory:", time.time() - start)
            memory.reset()

            # Attention
            start = time.time()
            attention(x)
            parallel_attention_times.append(time.time() - start)
            print("Attention:", time.time() - start)

        print()

        # Sequential Time
        memory.parallelize = False
        memory = memory.cpu()
        attention = attention.cpu()
        print("Sequential:")
        for seq_len in sequential_seq_lengths:
            print("seq_lengths:", seq_len)
            x = torch.randn(batch_size, seq_len, embed_dim)

            # Memory
            start = time.time()
            for i in range(x.size(1)):
                memory(x[:, i:i+1])
            sequential_memory_times.append(time.time() - start)
            print("Memory:", time.time() - start)
            memory.reset()

            # Attention
            start = time.time()
            for i in range(x.size(1)):
                attention(x[:, :i+1])
            sequential_attention_times.append(time.time() - start)
            print("Attention:", time.time() - start)

        # Plotting
        plt.figure(figsize=(12, 6))

        # Sequential inference
        plt.subplot(1, 2, 1)
        plt.plot(sequential_seq_lengths, sequential_memory_times, label='Memory (Sequential)', marker='o')
        plt.plot(sequential_seq_lengths, sequential_attention_times, label='Attention (Sequential)', marker='s')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (seconds)')
        plt.title('Sequential Inference')
        plt.legend()
        plt.grid()

        # Parallel inference
        plt.subplot(1, 2, 2)
        plt.plot(parallel_seq_lengths, parallel_memory_times, label='Memory (Parallel)', marker='o')
        plt.plot(parallel_seq_lengths, parallel_attention_times, label='Attention (Parallel)', marker='s')
        plt.xlabel('Sequence Length')
        plt.ylabel('Time (seconds)')
        plt.title('Parallel Inference')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

