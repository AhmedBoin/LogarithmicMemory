from typing import List

import torch
from torch import Tensor

from conv_summarizer import ConvSummarizer
from ..base_memory import BaseLogarithmicMemory


class LogarithmicMemory(BaseLogarithmicMemory):
    def __init__(self, embedding: int, heads: int = 1, bank: int = 1, depth_wise=True, parallelize=True):
        """
        Initializes the LogarithmicMemory module.
        """
        super().__init__(embedding, heads, bank, parallelize)
        self.summarizer = ConvSummarizer(embedding, bank, depth_wise)
    
    def parallel(self, x: Tensor) -> Tensor: # [batch, seq, embedding]
        """
        Processes input in parallel based on the logarithmic memory addresses.

        This method is a placeholder for a parallel memory access scheme.
        It's intended to define how an input `x` is processed when interacting
        with the memory in a parallel fashion, using logarithmic addresses.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
           Tensor: Output tensor of shape (batch_size, seq_len, 1+floor(log(seq_len))*bank, embedding_dim).
        """
        # Graph
        x_original = x
        x = x.unsqueeze(1).repeat(1, self.bank, 1, 1)
        batch_x, bank_x, seq_x, embed_x = x.shape

        x_last = x
        x_t = [x]
        for i in range(1, seq_x.bit_length()): # 1+floor(log(seq_x))
            # set length to be even and store shapes
            x_last = x_last[:, :, :-1, :] if x_last.size(-2) % 2 == 1 else x_last 
            batch, bank, seq, embedding = x_last.shape

            # concatenate each two tokens and calculate the tree graph
            x_last = self.summarizer(x_last)  # [batch, bank, seq/2, embedding]

            # sync shape of sequence by repeating.
            # handle zero padding values and extra cut length (extra non-power 2 index)
            x_temp = x_last.unsqueeze(-2).repeat(1, 1, 1, 2**i, 1).view(batch, bank, seq*2**(i-1), embedding) # [batch, bank, seq_x, embedding]

            exceed_shape = -(2**i-1)+(seq_x-seq*2**(i-1)) # = (seq_x+1) - 2**(i-1) * (seq + 2)
            exceed_shape = seq_x + 1 if exceed_shape == 0 else exceed_shape

            x_temp = torch.cat([
                torch.zeros(batch_x, bank_x, 2**i-1, embed_x, device=x.device), 
                x_temp[:, :, :exceed_shape, :]
                ], dim=-2) # [batch, bank_x, seq_x, embedding]
            
            x_t.append(x_temp)

        if len(x_t) == 1:
            return x_original.unsqueeze(-2)

        x = torch.stack(x_t[1:], dim=-2) # [batch, bank, seq, log(seq)-1, embedding]
        x = x.permute(0, 2, 3, 1, 4).contiguous() # [batch, seq, log(seq)-1, bank, embedding]
        x = x.view(batch_x, seq_x, -1, embed_x) # [batch, seq, log(seq)*bank-1, embedding]
        x = torch.cat([x_original.unsqueeze(-2), x], dim=-2) # [batch, seq, log(seq)*bank, embedding]
        return x

    def sequential(self, x: Tensor) -> Tensor: # [batch, seq, embedding]
        """
        Processes input sequentially, writing to and reading from the memory based on
        logarithmic addressing.

        This method is a placeholder for a sequential memory access scheme. 
        It defines how an input `x` interacts with the memory one step at a time,
        using logarithmic addresses.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 1, embedding_dim).

        Returns:
           Tensor: Output tensor of shape (batch_size, 1, 1+floor(log(seq_len))*bank, embedding_dim).
        """
        # Memory
        x_original = x
        x = x.unsqueeze(1).repeat(1, self.bank, 1, 1) # [batch, bank, seq, embedding]
        batch_x, bank_x, seq_x, embed_x = x.shape

        new_memory = [x]
        for i, summarize in enumerate(self.summarize_table()):
            # summarize (update)
            if summarize:
                combination = self.summarizer(torch.cat([self.memory[i], new_memory[i]], dim=2))
                new_memory.append(combination)

            # copy
            else:
                new_memory.append(self.memory[i + 1])
        self.memory = new_memory

        if len(new_memory) == 1:
            return x_original.unsqueeze(-2)
            
        x = torch.stack(new_memory[1:], dim=-2) # [batch, bank, seq, log(seq)-1, embedding]
        x = x.permute(0, 2, 3, 1, 4).contiguous() # [batch, seq, log(seq)-1, bank, embedding]
        x = x.view(batch_x, seq_x, -1, embed_x) # [batch, seq, log(seq)*bank-1, embedding]
        x = torch.cat([x_original.unsqueeze(-2), x], dim=-2) # [batch, seq, log(seq)*bank, embedding]
        return x
    
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)


if __name__ == "__main__":
    import time
    with torch.no_grad():
        batch_size = 1
        seq_len = 128
        embed_dim = 64
        heads = 8
        memory_banks = 8

        model = LogarithmicMemory(embed_dim, heads, memory_banks, False)#.to("mps")
        print("Parameters:", sum([param.numel() for param in model.parameters()]))
        model: LogarithmicMemory = torch.compile(model)
        x = torch.randn(batch_size, seq_len, embed_dim)#.to("mps")

        """
        inference speed test
        """
        print("inference speed test:")
        print("parallel inference:")
        start = time.time()
        for i in range(x.size(1)):
            model.attention(model.parallel(x[:, 0:i+1]))
        print("parallel time:", time.time()-start)

        print()
        print("sequential inference:")
        start = time.time()
        for i in range(x.size(1)):
            model.attention(model.sequential(x[:, i:i+1]))
        print("parallel time:", time.time()-start)


        """
        value check test
        """
        model.reset()
        x_parallel = model.attention(model.parallel(x))
        for i in range(x.size(1)):
            x_sequential = model.attention(model.sequential(x[:, i:i+1]))
            print(x_parallel[:, i].allclose(x_sequential[:, 0], 1e-2))
