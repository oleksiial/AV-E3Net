import torchaudio
from data.datamodule import DataModule
from utils.plot_waveforms import plot_waveforms
from ave3net.model import AVE3Net
# from ave3net.model_AO import AVE3Net
import torch
from torchvision import transforms
import time
import torch.nn.functional as F
import torchvision

clean1, _sr = torchaudio.load('/data/LRS3_30h/test/clean/YD5PFdghryc/00001.wav')
clean2, _sr = torchaudio.load('/data/LRS3_30h/test/clean/BhMKmovNjvc/00001.wav') # 5s

clean1 = clean1[:, :16000*5]
clean2 = clean2[:, :16000*5]

torchaudio.save("mix.wav", clean1 + clean2 * 1, 16000)
