from typing import Tuple
import utils.logger as logger
import torchvision
import torchaudio
import torch
from torchdata.datapipes.iter import FileLister

# dp = FileLister(root="/data/LRS3-kai-2sp/test/mix")
# print(list(dp))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, audio_only=False):  # split train - valid - test
        self.logger = logger.get_logger(self.__class__.__name__, logger.logging.NOTSET)
        self.ROOT = '/data/LRS3-kai-2sp'
        self.split = split
        self.audio_only = audio_only

        self.mix = list(FileLister(root=(self.ROOT + '/' + self.split + '/mix')))
        self.clean_s1 = list(FileLister(root=(self.ROOT + '/' + self.split + '/s1')))
        self.clean_s2 = list(FileLister(root=(self.ROOT + '/' + self.split + '/s2')))

        self.mix = self.mix + self.mix
        self.clean = self.clean_s1 + self.clean_s2

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.logger.debug('__getitem__ start')

        noisy_path = self.mix[i]
        clean_path = self.clean[i]
        
        clean_waveform, _sr = torchaudio.load(clean_path)
        noisy_waveform, _sr = torchaudio.load(noisy_path)

        self.logger.debug(f'clean_waveform.shape: {clean_waveform.shape}')  # clean_waveform.shape: [1, 51200]
        self.logger.debug(f'noisy_waveform.shape: {noisy_waveform.shape}')  # noisy_waveform.shape: [1, 51200]

        if self.audio_only:
            return noisy_waveform, clean_waveform

        speaker_id = noisy_path.split('_')[0].split('/')[-1] if "/s1/" in clean_path else noisy_path.split('_')[3]
        video_id = noisy_path.split('_')[1] if "/s1/" in clean_path else noisy_path.split('_')[4]

        video_path = self.ROOT + '/' + self.split + '/roi/' + speaker_id + '/' + video_id + '.mp4'

        self.logger.debug(f'noisy_path: {noisy_path}')
        self.logger.debug(f'clean_path: {clean_path}')
        self.logger.debug(f'video_path: {video_path}')


        vframes, _aframes, _info = torchvision.io.read_video(video_path, pts_unit='sec', output_format='TCHW')  # pts_init to avoid warning
        self.logger.debug(f'vframes.shape: {vframes.shape}')
        vframes = vframes / 255

        self.logger.debug('__getitem__ end')

        return vframes, noisy_waveform, clean_waveform

    def __len__(self):
        return len(self.mix)


if __name__ == '__main__':
    dataset = Dataset('test')
    vframes, noisy, clean, meta = dataset[2500]
    print(len(dataset))
    print(meta[0])
    print(meta[1])
    print(meta[2])
