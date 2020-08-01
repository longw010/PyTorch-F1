import os
from pathlib import Path
import pytest
from ..info import info_str
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)

def test_info_str():
    print(info_str(resnet18, 1, 128))

