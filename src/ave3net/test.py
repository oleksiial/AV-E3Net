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


datamodule = DataModule()
datamodule.setup("test")

test_dataloader = datamodule.test_dataloader()
model = AVE3Net.load_from_checkpoint(
    checkpoint_path="lightning_logs/version_177/checkpoints/checkpoint.ckpt",
    map_location=torch.device('cpu')
)
# model = AVE3Net()

# /data/LRS3_30h/test/clean/YD5PFdghryc/00001.wav
# /data/LRS3_30h/test/noisy/YD5PFdghryc/00001.wav
# /data/LRS3_30h/test/video/YD5PFdghryc/00001.mp4

# vframes, noisy, clean = test_dataloader.dataset[205]

clean, _sr = torchaudio.load('/data/LRS3_30h/test/clean/YD5PFdghryc/00001.wav')
clean = clean[:, :16000*5]

noisy, _sr = torchaudio.load('/export/home/7alieksi/ba/mix.wav')

vframes, _aframes, _info = torchvision.io.read_video('/data/LRS3_30h/test/video/YD5PFdghryc/00001.mp4', pts_unit='sec', output_format='TCHW')
vframes = vframes[:25*5]  # pts_init to avoid warning
vframes = vframes / 255

print(f'collate output: {vframes.shape} {noisy.shape} {clean.shape} {vframes.type()}')


# vframes, noisy, clean = test_dataloader.dataset[315]

def process_with_rtf(model: AVE3Net, vframes: torch.Tensor, noisy: torch.Tensor):
    print('process_with_rtf input', vframes.shape, noisy.shape)

    # 640 samples = 40ms = 1 video frame. 16kHz/25fps
    # 160 samples = 10ms = 0.25 video frame. Every frame processed 4 times.


    a = noisy.split(640, 1)
    # a = noisy.split(160, 1)[:-1]  # discard last incomplete chunk
    v = F.interpolate(vframes.transpose(0, 3), (vframes.size(-1), len(a))).transpose(0, 3)
    v = v.split(1)
    v = [*v, v[-1]]

    print(len(a), len(v))

    rtfs = torch.Tensor([])
    for i in range(1):
        start = time.time()
        x_hat = torch.tensor([])
        for ach, vch in zip(a, v):
            x_hat_ch = model((vch.unsqueeze(0), ach))
            x_hat = torch.cat((x_hat, x_hat_ch[0][0]))
        x_hat = x_hat.unsqueeze(0)
        # x_hat = model((vframes, noisy)).detach()[0]
        # print(x_hat.shape)
        end = time.time()
        # x_hat = x_hat.detach()
        elapsed = end - start
        audio_duration = noisy.size(1) / 16000
        rtf = elapsed / audio_duration
        rtfs = torch.cat((rtfs, torch.Tensor([rtf])))
    return x_hat, rtfs.mean()


# x_hat, rtf = process_with_rtf(model, vframes, noisy)
# x_hat = x_hat.detach()
# print('rtf', rtf)

x_hat = model((vframes, noisy))
x_hat = x_hat.detach()[0]

torchaudio.save("x_hat.wav", x_hat, 16000)
torchaudio.save("clean.wav", clean, 16000)
torchaudio.save("noisy.wav", noisy, 16000)
# torchaudio.save("clean.wav", clean, 16000)
# torchaudio.save("noisy.wav", a, 16000)

plot_waveforms(16000, [(noisy, 'mix'), (x_hat, 'estimated'), (clean, 'clean')])


print('done')
