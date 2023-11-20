from typing import Tuple
import utils.logger as logger
import torchvision
import torchaudio
import torch
from torchdata.datapipes.iter import FileLister

# dp = FileLister(root="/data/LRS3-kai-2sp/test/mix")
# print(list(dp))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, target_speaker, audio_only=False):  # split train - valid - test
        self.logger = logger.get_logger(self.__class__.__name__, logger.logging.NOTSET)
        self.ROOT = '/data/LRS3-kai-2sp'
        self.split = split
        self.audio_only = audio_only

        self.mix = list(FileLister(root=(self.ROOT + '/' + self.split + '/mix')))
        self.clean = list(FileLister(root=(self.ROOT + '/' + self.split + '/' + target_speaker)))

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noisy_path = self.mix[i]
        clean_path = self.clean[i]
        
        clean_waveform, _sr = torchaudio.load(clean_path)
        noisy_waveform, _sr = torchaudio.load(noisy_path)

        self.logger.debug(f'clean_waveform.shape: {clean_waveform.shape}')  # clean_waveform.shape: [1, 51200]
        self.logger.debug(f'noisy_waveform.shape: {noisy_waveform.shape}')  # noisy_waveform.shape: [1, 51200]

        if self.audio_only:
            return noisy_waveform, clean_waveform

        speaker_id = noisy_path.split('_')[0].split('/')[-1]
        video_id = noisy_path.split('_')[1]

        video_path = self.ROOT + '/' + self.split + '/roi/' + speaker_id + '/' + video_id + '.mp4'

        vframes, _aframes, _info = torchvision.io.read_video(
            video_path, pts_unit='sec', output_format='TCHW')  # pts_init to avoid warning
        
        self.logger.debug(f'vframes.shape: {vframes.shape}')  # vframes.shape: [80, 3, 96, 96]
        # normalization for shufflenet https://pytorch.org/hub/pytorch_vision_shufflenet_v2/
        # preprocess = torchvision.transforms.Compose([
        #     torchvision.transforms.Resize(224),
        #     torchvision.transforms.CenterCrop(224),
        #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        # vframes = preprocess(vframes)
        vframes = vframes / 255
        return vframes, noisy_waveform, clean_waveform  

    def __len__(self):
        return len(self.mix)


if __name__ == '__main__':
    dataset = Dataset('train', 's1', True)
    noisy, clean = dataset[20]
