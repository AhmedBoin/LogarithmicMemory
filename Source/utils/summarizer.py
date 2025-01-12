import torch
from torch import nn, Tensor


class Summarizer(nn.Module):
    def __init__(self, embedding: int, bank: int = 1):
        super(Summarizer, self).__init__()
        self.weight = nn.Parameter(torch.randn(bank, 2 * embedding, embedding))
        self.bias = nn.Parameter(torch.randn(bank, 1, embedding))
        nn.init.xavier_normal_(self.weight)
        nn.init.xavier_normal_(self.bias)
        self.weight.data = self.weight.data * 0.1
        self.bias.data = self.bias.data * 0.1

    def forward(self, x: Tensor): # [..., bank, seq/2, embedding*2]
        return torch.einsum("...ijk,ikl->...ijl", x, self.weight) + self.bias
    
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)
    

if __name__ == "__main__":
    embedding = 64
    sequence = 32
    bank = 2
    model = Summarizer(embedding, bank)
    print(f"model parameters: {sum([param.numel() for param in model.parameters()])}")

    input_tokens = torch.randn(1, bank, sequence//2, 2*embedding)  # (Batch, Bank, Sequence Length / 2, Embedding * 2)
    output_tokens = model(input_tokens)
    print(input_tokens.shape, "->", output_tokens.shape)
    
