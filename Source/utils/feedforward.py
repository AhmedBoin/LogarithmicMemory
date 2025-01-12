from torch import nn, Tensor


class FeedForward(nn.Module):
    def __init__(self, embedding: int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embedding, 4*embedding)
        self.fc2 = nn.Linear(4*embedding, embedding)
        self.activation = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.activation(self.fc1(x)))
    
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)
    
