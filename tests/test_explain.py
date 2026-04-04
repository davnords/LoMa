import torch
import torch._dynamo as dynamo

from loma.device import device
from loma.loma import LoMa

model = LoMa(LoMa.Cfg(compile=False)).to(device)

explanation = dynamo.explain(model)(
    torch.randn(1, 2048, 2).to(device),
    torch.randn(1, 2048, 2).to(device),
    torch.randn(1, 2048, 256).to(device),
    torch.randn(1, 2048, 256).to(device)
)
print(explanation)
