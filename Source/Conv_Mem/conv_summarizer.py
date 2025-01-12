from torch import nn, Tensor


class ConvSummarizer(nn.Module):
    def __init__(self, embedding: int, bank: int = 1, depth_wise=True):
        super(ConvSummarizer, self).__init__()
        groups = bank if depth_wise else 1
        self.conv = nn.Conv1d(embedding*bank, embedding*bank, kernel_size=2, stride=2, groups=groups)

    def forward(self, x: Tensor): # [batch, bank, seq, embedding]
        batch, bank, seq, embedding = x.shape
        x = x.transpose(-1, -2).reshape(batch, bank*embedding, seq)
        x = self.conv(x)
        x = x.view(batch, bank, embedding, -1).transpose(-1, -2)
        return x
    
