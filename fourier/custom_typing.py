import torch

import typing

DeviceLike = typing.Union[str, torch.device]

FPTensor = typing.Union[torch.FloatTensor, torch.DoubleTensor]
