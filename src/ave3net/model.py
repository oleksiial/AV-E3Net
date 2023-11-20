import lightning.pytorch as pl
import torch
from torchinfo import summary
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.save_fig import save_fig
from typing import List, Tuple
from ave3net.shufflenet_encoder import _shufflenetv2_05
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, perceptual_evaluation_speech_quality, signal_distortion_ratio
import utils.logger as logger


class ProjectionBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.PReLU(),
            nn.LayerNorm(out_features)
        )

    def forward(self, x):
        return self.block(x)


class GSFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.projection_block = ProjectionBlock(512, 512)
        self.gate = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        self.layernorm = nn.LayerNorm(512)

    def forward(self, x, dense_audio):
        vx, ax = x
        # print('GS input', 'audio', ax.shape, 'video', vx.shape)
        # vx = F.interpolate(vx.transpose(1, 2), ax.size(1)).transpose(1, 2)
        vx = torch.cat([ax, vx], dim=2)
        vx = self.gate(vx)
        dense_audio += ax
        ax *= vx
        ax = self.projection_block(ax)
        ax += dense_audio
        ax = self.layernorm(ax)
        # print('GS output', ax.shape)
        return ax, dense_audio


class LSTMBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.gsfusion = GSFusion()

        self.map_to_high_dim_a = nn.Sequential(nn.Linear(512, 1024), nn.PReLU(),
                                               nn.Linear(1024, 512), nn.PReLU(), nn.LayerNorm(512))
        self.map_to_high_dim_v = nn.Sequential(nn.Linear(512, 1024), nn.PReLU(),
                                               nn.Linear(1024, 512), nn.PReLU(), nn.LayerNorm(512))

        self.lstm_a = nn.LSTM(512, 512)
        self.layer_norm_a = nn.LayerNorm(512)
        self.layer_norm_a2 = nn.LayerNorm(512)
        self.lstm_v = nn.LSTM(512, 512)
        self.layer_norm_v = nn.LayerNorm(512)
        self.layer_norm_v2 = nn.LayerNorm(512)

    def forward(self, x, dense_audio, dense_video, ha, hv):
        vx, ax = x
        ax, dense_audio = self.gsfusion((vx, ax), dense_audio)

        ax = self.map_to_high_dim_a(ax)
        dense_audio += ax
        if (ha != None):
            ax, ha = self.lstm_a(ax, ha)
        else:
            ax, ha = self.lstm_a(ax)

        ax = self.layer_norm_a(ax)
        ax += dense_audio
        ax = self.layer_norm_a2(ax)

        vx = self.map_to_high_dim_v(vx)
        dense_video += vx
        if hv != None:
            vx, hv = self.lstm_v(vx, hv)
        else:
            vx, hv = self.lstm_v(vx)
        vx = self.layer_norm_v(vx)
        vx += dense_video
        vx = self.layer_norm_v2(vx)

        return (vx, ax), dense_audio, dense_video, ha, hv


class AVE3NetModule(nn.Module):
    def __init__(self, fusion_block, n_lstm=4):
        super().__init__()
        self.debug_logger = logger.get_logger(self.__class__.__name__, logger.logging.DEBUG)

        self.ha = None
        self.hv = None

        # audio
        self.window = 320
        self.stride = 160
        self.encoder = nn.Conv1d(1, 2048, kernel_size=self.window, stride=self.stride)
        self.l1 = nn.Sequential(nn.PReLU(), nn.LayerNorm(2048))
        self.audio_projection_block = ProjectionBlock(2048, 512)
        self.mask_prediction = nn.Sequential(nn.Linear(512, 2048), nn.Sigmoid())
        self.decoder = nn.ConvTranspose1d(2048, 1, self.window, self.stride)

        # video
        self.shufflenet = _shufflenetv2_05()
        self.video_projection_block = ProjectionBlock(1024, 512)

        # fusion
        self.gsfusion = fusion_block

        # LSTM
        self.lstm_blocks = nn.ModuleList()
        for _ in range(n_lstm):
            self.lstm_blocks.append(LSTMBlock())

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        self.debug_logger.debug(f'pad_signal input {input.shape}')
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.window - (self.stride + nsample %
                              self.window) % self.window
        
        self.debug_logger.debug(f'batch_size {batch_size}')
        self.debug_logger.debug(f'nsample {nsample}')
        self.debug_logger.debug(f'rest {rest}')
        self.debug_logger.debug(f'self.stride {self.stride}')
        self.debug_logger.debug(f'self.window {self.window}')


        # if rest > 0:
        #     pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
        #     self.debug_logger.debug(f'pad {pad.shape}')

        #     input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(
            batch_size, 1, self.stride)).type(input.type())

        self.debug_logger.debug(f'pad_aux {pad_aux.shape}')

        input = torch.cat([pad_aux, input, pad_aux], 2)

        self.debug_logger.debug(f'pad_signal output/rest {input.shape}/{rest}')


        # return input, rest
        return input, 0

    # def on_after_backward(self) -> None:
    #     print("on_before_opt enter")
    #     for name, p in self.named_parameters():
    #         if p.grad is None:
    #             print(name)
    #     print("on_before_opt exit")

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        """
        x of shape (vx, ax)

        vx 5D (batch_size, frames, channels, width, height) or 4D without batch_size

        ax 3D (batch_size, channels=1, samples) or 2D without batch_size
        """

        vx, ax = x
        self.debug_logger.debug(f'forward input vx.shape {vx.shape}, ax.shape {ax.shape}')

        if vx.dim() not in [4, 5]:
            raise RuntimeError(f"AV-E3Net video input wrong shape: {vx.shape}")

        if ax.dim() not in [2, 3]:
            raise RuntimeError(f"AV-E3Net audio input wrong shape: {ax.shape}")

        # add minibatch dim to inputs
        if vx.dim() == 4:
            vx = vx.unsqueeze(0)
        if ax.dim() == 2:
            ax = ax.unsqueeze(0)

        # audio

        ax, rest = self.pad_signal(ax)
        audio_encoded = self.encoder(ax)
        self.debug_logger.debug(f'audio_encoded.shape {audio_encoded.shape}')

        # outmap_min, _ = torch.min(audio_encoded, dim=1, keepdim=True)
        # outmap_max, _ = torch.max(audio_encoded, dim=1, keepdim=True)
        # ae = (audio_encoded - outmap_min) / (outmap_max - outmap_min)
        # ae = ae.view(-1, 64)
        # print(ae.size())
        # ae = ae.detach().numpy()
        # # print(audio_encoded.size())
        # save_fig('audio_encoded.png', ae)

        ax = audio_encoded.transpose(1, 2)
        ax = self.l1(ax)
        self.debug_logger.debug(f'ax.shape {ax.shape}')
        ax = self.audio_projection_block(ax)
        self.debug_logger.debug(f'ax.shape {ax.shape}')

        ##############

        # video

        n_frames = vx.size(1)
        vx = vx.view(-1, 3, 96, 96)  # [4, 71, 3, 96, 96] to [284, 3, 96, 96]
        vx = self.shufflenet(vx)
        vx = vx.view(-1, n_frames, 1024)  # transform back
        vx = self.video_projection_block(vx)
        # print('after projection blocks', 'audio', ax.shape, 'video', vx.shape)
        ##############

        # LSMT blocks

        # upsample video
        self.debug_logger.debug(f'vx.shape/ax.shape {vx.shape}/{ax.shape} --- {vx[0].mean(1)}')
        vx = F.interpolate(vx.transpose(1, 2), ax.size(1)).transpose(1, 2)
        self.debug_logger.debug(f'vx.shape/ax.shape {vx.shape}/{ax.shape} --- {vx[0].mean(1)}')
        # initialize dense sums
        dense_audio = torch.zeros_like(ax)
        dense_video = torch.zeros_like(vx)

        ha, hv = self.ha, self.hv
        for lstm in self.lstm_blocks:
            (vx, ax), dense_audio, dense_video, ha, hv = lstm((vx, ax), dense_audio, dense_video, ha, hv)
        # self.ha, self.hv = ha, hv

        ##############

        # final stage
        ax, dense_audio = self.gsfusion((vx, ax), dense_audio)
        ax = self.mask_prediction(ax)
        ax = ax.transpose(1, 2)

        # print('after sigmoid', ax.shape)

        ax = ax * audio_encoded

        ax = self.decoder(ax)
        ax = ax[:, :, self.stride: -(rest + self.stride)].contiguous()  # B*C, 1, L

        ##############

        return ax


class AVE3Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.debug_logger = logger.get_logger(self.__class__.__name__, logger.logging.NOTSET)
        self.ave3net = AVE3NetModule(GSFusion(), 4)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        return self.ave3net(x)

    def process_batch(self, batch: Tuple[List[Tensor], List[Tensor], List[Tensor]], batch_idx):
        video, noisy, clean = batch
        self.debug_logger.debug(f'process_batch input video[0]{video[0].shape}, noisy[0]{noisy[0].shape}')

        # convert audios from [1, T] to [T, 1] for pad_sequence
        noisy = [x.transpose(0, 1) for x in noisy]
        clean = [x.transpose(0, 1) for x in clean]

        # pad batch to max size
        # audios transposed back to [1, T] after padding
        video = nn.utils.rnn.pad_sequence(video, batch_first=True)
        noisy = nn.utils.rnn.pad_sequence(noisy, batch_first=True).transpose(1, 2)
        clean = nn.utils.rnn.pad_sequence(clean, batch_first=True).transpose(1, 2)
        # padded video [16, 153, 3, 96, 96], noisy [16, 1, 98304], clean [16, 1, 98304]
        self.debug_logger.debug(f'padded video {video.shape}, noisy {noisy.shape}, clean {clean.shape}')

        x_hat = self.forward((video, noisy))
        return x_hat, clean

    def training_step(self, batch: Tuple[List[Tensor], List[Tensor], List[Tensor]], batch_idx):
        self.ha = None
        self.hv = None

        x_hat, clean = self.process_batch(batch, batch_idx)
        loss = nn.functional.mse_loss(x_hat, clean)
        self.log("train_loss", loss, prog_bar=True, batch_size=16)
        return loss

    # def validation_step(self, batch: Tuple[List[Tensor], List[Tensor], List[Tensor]], batch_idx):
    #     self.ha = None
    #     self.hv = None

    #     x_hat, clean = self.process_batch(batch, batch_idx)
    #     loss = nn.functional.mse_loss(x_hat, clean)
    #     self.log("validation_loss", loss, prog_bar=True, sync_dist=True, batch_size=16)
    #     return loss

    def test_step(self, batch: Tuple[List[Tensor], List[Tensor], List[Tensor]], batch_idx):
        self.ha = None
        self.hv = None

        x_hat, clean = self.process_batch(batch, batch_idx)

        pesq = perceptual_evaluation_speech_quality(x_hat, clean, 16000, 'wb')
        sdr = signal_distortion_ratio(x_hat, clean)
        sisdr = scale_invariant_signal_distortion_ratio(x_hat, clean)
        loss = nn.functional.mse_loss(x_hat, clean)

        log = {
            "loss": loss,
            "pesq": pesq.mean(),
            "sdr": sdr.mean(),
            "sisdr": sisdr.mean()
        }

        self.log_dict(log, prog_bar=True, batch_size=16)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    data = [(torch.rand((16, 25, 3, 96, 96)), torch.rand((16, 1, 16000)))]
    summary(AVE3Net(), input_data=data, col_names=["input_size",
                                                   "output_size",
                                                   "num_params"], depth=2)
    # summary(AVE3Net(), input_size=(1, 60000))
