import torch
from torch import nn, Tensor


class Expander(nn.Module):
    def __init__(self, embedding, expand=1):
        super(Expander, self).__init__()
        assert expand >= 0, "Expand value must be greater than or equal to 0."
        self.expand = expand
        self.conv_transpose = nn.ConvTranspose1d(in_channels=embedding, out_channels=embedding,  kernel_size=1+expand)

    def forward(self, x: Tensor): # [batch, expand, bank, seq, embedding]
        batch, expand, bank, seq, embedding = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(batch*bank*seq, embedding, expand)
        x = self.conv_transpose(x)
        x = x.view(batch, bank, seq, embedding, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x # [batch, expand, bank, seq, embedding]
    
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)


if __name__ == "__main__":
    embedding = 32
    expand = 2
    model = Expander(embedding, expand)
    print(f"model parameters: {sum([param.numel() for param in model.parameters()])}")
    
    for i in range(1, 10):
        input_tokens = torch.randn(1, i, 1, 1, embedding)  # [batch, expand, bank, seq, embedding]
        output_tokens = model(input_tokens)
        print(input_tokens.shape, "->", output_tokens.shape)
