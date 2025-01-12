from typing import List

from torch import nn, Tensor

from utils.summarizer import Summarizer
from utils.single_vector_attention import SingleVectorAttention
from utils.feedforward import FeedForward


class BaseLogarithmicMemory(nn.Module):
    """
    A base module for exploring memory access with a logarithmic addressing scheme.

    This module provides a framework for experimenting with different methods 
    of interacting with a memory structure. It defines a logarithmic memory
    pointer and several methods to access this memory, intended for research
    and experimentation purposes. It offers parallel, sequential, and attention-based
    methods of accessing the memory.
    """
    def __init__(self, embedding: int, heads: int = 1, bank: int = 1, parallelize=True):
        """
        Initializes the BaseLogarithmicMemory module.
        """
        super().__init__()
        self.bank = bank
        self.parallelize = parallelize

        self.norm1 = nn.RMSNorm(embedding)
        self.norm2 = nn.RMSNorm(embedding)
        self.summarizer = Summarizer(embedding, bank)
        self.sva = SingleVectorAttention(embedding, heads)
        self.feedforward = FeedForward(embedding)

        self.reset()

    def reset(self):
        """
        Resets the internal memory and the memory pointer.

        This function sets the internal memory (`self.memory`) to `None`, 
        and the memory pointer (`self.ptr`) to 0. This allows for starting from
        a clean slate in each new experiment.
        """
        self.memory = None
        self.ptr = 0

    def summarize_table(self) -> List[bool]:
        """
        Summarizes a memory table based on a logarithmic addressing scheme.

        This function generates a logarithmic address by creating a sequence of 
        binary differences between the current memory pointer (`self.ptr`) and
        the next one (`self.ptr + 1`). The memory pointer will also be incremented
        in the process.

        Returns:
            list: A list of boolean values representing the bit-wise differences between
                  consecutive memory pointers.
        """
        a, b = self.ptr, self.ptr+1
        self.ptr = b
        max_len = max(a.bit_length(), b.bit_length())
        return [(a ^ b) & (1 << i) != 0 for i in range(1, max_len)]
    
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
        raise NotImplementedError("This method is not implemented yet.")

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
        raise NotImplementedError("This method is not implemented yet.")
    
    def attention(self, x: Tensor) -> Tensor:
        return self.sva(x)
    
    def forward(self, x: Tensor) -> Tensor: # [batch_size, seq_len, embed_dim]
        """
        Forward pass that selects the desired memory processing method.

        This method is a placeholder for selecting a concrete memory processing
        scheme to be applied to input `x`. It's meant to choose from different
        access methods like `parallel` or `sequential`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
           Tensor: Output tensor of shape (batch_size, seq_len, embedding_dim).
        """
        x_ = self.norm1(x)
        x_ = self.parallel(x_) if self.parallelize else self.sequential(x_)
        x = x + self.attention(x_)
        x_ = self.norm2(x)
        x = x + self.feedforward(x_)
        return x
    
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)
    
