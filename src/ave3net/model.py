import time
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torchinfo import summary
from torch import Tensor, nn
import torch.nn.functional as F
from torch.autograd import Variable
from data.datamodule import DataModule
from utils.save_fig import save_fig
from utils.plot_waveforms import plot_waveforms
from typing import List, Tuple
from ave3net.shufflenet_encoder import _shufflenetv2_05
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, perceptual_evaluation_speech_quality, signal_distortion_ratio
import utils.logger as logger
from torchaudio.transforms import AmplitudeToDB


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
        vx = torch.cat([ax, vx], dim=2)
        vx = self.gate(vx)
        dense_audio = dense_audio + ax
        ax = ax * vx
        ax = self.projection_block(ax)
        ax = ax + dense_audio
        ax = self.layernorm(ax)
        # print('GS output', ax.shape)
        return ax, dense_audio


class LSTMBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.gsfusion = GSFusion()

        self.map_to_high_dim_a = nn.Sequential(
            nn.Linear(512, 1024), nn.PReLU(),
            nn.Linear(1024, 512), nn.PReLU(),
            nn.LayerNorm(512)
        )

        self.map_to_high_dim_v = nn.Sequential(
            nn.Linear(512, 1024), nn.PReLU(),
            nn.Linear(1024, 512), nn.PReLU(),
            nn.LayerNorm(512)
        )

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
        dense_audio = dense_audio + ax
        ax, ha = self.lstm_a(ax) if ha == None else self.lstm_a(ax, ha)
        ax = self.layer_norm_a(ax)
        ax = ax + dense_audio
        ax = self.layer_norm_a2(ax)

        vx = self.map_to_high_dim_v(vx)
        dense_video = dense_video + vx
        vx, hv = self.lstm_v(vx) if hv == None else self.lstm_v(vx, hv)
        vx = self.layer_norm_v(vx)
        vx += dense_video
        vx = self.layer_norm_v2(vx)

        return (vx, ax), dense_audio, dense_video, ha, hv


class AVE3NetModule(nn.Module):
    def __init__(self, fusion_block, n_lstm=4):
        super().__init__()
        self.debug_logger = logger.get_logger(self.__class__.__name__, logger.logging.NOTSET)

        self.ha = [None] * n_lstm
        self.hv = [None] * n_lstm

        # audio
        self.window = 320
        self.stride = 160
        self.encoder = nn.Conv1d(1, out_channels=2048, kernel_size=self.window, stride=self.stride)
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
        nsamples = input.size(2)
        rest = self.window - (self.stride + nsamples % self.window) % self.window
        
        self.debug_logger.debug(f'batch_size {batch_size}')
        self.debug_logger.debug(f'nsample {nsamples}')
        self.debug_logger.debug(f'rest {rest}')
        self.debug_logger.debug(f'self.stride {self.stride}')
        self.debug_logger.debug(f'self.window {self.window}')

        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            self.debug_logger.debug(f'pad {pad.shape}')
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        self.debug_logger.debug(f'pad_aux {pad_aux.shape}')
        input = torch.cat([pad_aux, input, pad_aux], 2)

        self.debug_logger.debug(f'pad_signal output/rest {input.shape}/{rest}')
        return input, rest

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
        self.debug_logger.debug('forward start')

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
        # audio_encoded = torch.stft(ax, n_fft = 512, win_length=self.window, hop_length=self.stride, return_complex=True)
        self.debug_logger.debug(f'audio_encoded.shape {audio_encoded.shape}')

        ax = audio_encoded.transpose(1, 2)
        ax = self.l1(ax)
        self.debug_logger.debug(f'ax.shape after encoding and l1 {ax.shape}')
        ax = self.audio_projection_block(ax)
        self.debug_logger.debug(f'ax.shape after proj block {ax.shape}')

        ##############

        # video

        n_frames = vx.size(1)
        vx = vx.view(-1, 3, 96, 96)  # [4, 71, 3, 96, 96] to [284, 3, 96, 96]
        vx = self.shufflenet(vx)
        self.debug_logger.debug(f'vx.shape after shufflenet {vx.shape}')
        vx = vx.view(-1, n_frames, 1024)  # transform back
        vx = self.video_projection_block(vx)
        self.debug_logger.debug(f'vx.shape after shufflenet and proj block {vx.shape}')
        ##############

        # LSMT blocks

        # upsample video
        vx = F.interpolate(vx.transpose(1, 2), ax.size(1)).transpose(1, 2)
        self.debug_logger.debug(f'vx.shape/ax.shape after upsampling {vx.shape}/{ax.shape}')
        # initialize dense sums
        dense_audio = torch.zeros_like(ax)
        dense_video = torch.zeros_like(vx)

        # print(self.ha[0] is None)

        # ha, hv = self.ha, self.hv
        for i, lstm in enumerate(self.lstm_blocks):
            (vx, ax), dense_audio, dense_video, ha, hv = lstm((vx, ax), dense_audio, dense_video, self.ha[i], self.hv[i])
            self.ha[i] = ha
            self.hv[i] = hv
        # self.ha, self.hv = ha, hv
        self.debug_logger.debug(f'vx.shape/ax.shape after lsmts {vx.shape}/{ax.shape}')

        ##############

        # prediction
        ax, dense_audio = self.gsfusion((vx, ax), dense_audio)
        self.debug_logger.debug(f'ax.shape after last fusion {ax.shape}')
        ax = self.mask_prediction(ax)
        ax = ax.transpose(1, 2)
        self.debug_logger.debug(f'ax.shape after mask prediction {ax.shape}')

        ax = ax * audio_encoded
        ax = self.decoder(ax)
        self.debug_logger.debug(f'ax.shape after decoder {ax.shape}')
        ax = ax[:, :, self.stride: -(rest + self.stride)].contiguous()  # B*C, 1, L

        ##############

        ax = ax / torch.max(torch.abs(ax)) # normalize output to [-1; 1]

        self.debug_logger.debug(f'ax.mean {torch.mean(ax)}')
        self.debug_logger.debug(f'ax.min {torch.min(ax)}')
        self.debug_logger.debug(f'ax.max {torch.max(ax)}')
        self.debug_logger.debug('forward end')

        return ax


class AVE3Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.debug_logger = logger.get_logger(self.__class__.__name__, logger.logging.NOTSET)
        self.ave3net = AVE3NetModule(GSFusion(), 4)


    def forward(self, x: Tuple[Tensor, Tensor]) -> Tensor:
        # return x[1]
        return self.ave3net(x)

    def process_batch(self, batch: Tuple[List[Tensor], List[Tensor], List[Tensor]], batch_idx):
        video, noisy, clean = batch
        self.debug_logger.debug(f'process_batch input video[0]{video[0].shape}, noisy[0]{noisy[0].shape}')

        # convert audios from [1, T] to [T, 1] for pad_sequence
        # noisy = [x.transpose(0, 1) for x in noisy]
        # clean = [x.transpose(0, 1) for x in clean]

        # # video = torch.stack((video))
        # # noisy = torch.stack((noisy)).transpose(1, 2)
        # # clean = torch.stack((clean)).transpose(1, 2)

        # # pad batch to max size
        # # audios transposed back to [1, T] after padding
        # video = nn.utils.rnn.pad_sequence(video, batch_first=True)
        # noisy = nn.utils.rnn.pad_sequence(noisy, batch_first=True).transpose(1, 2)
        # clean = nn.utils.rnn.pad_sequence(clean, batch_first=True).transpose(1, 2)
        # padded video [16, 153, 3, 96, 96], noisy [16, 1, 98304], clean [16, 1, 98304]
        self.debug_logger.debug(f'padded video {video.shape}, noisy {noisy.shape}, clean {clean.shape}')

        x_hat = self.forward((video, noisy))
        return x_hat, clean
    
    def compute_loss(self, x_hat: Tensor, clean: Tensor) -> Tensor:

        # return compute_loss(x_hat, clean)

        sisdr = scale_invariant_signal_distortion_ratio(x_hat, clean)
        return -torch.mean(sisdr)
    
        # x_hat, x_hat_amp = compute_cprs_stft(x_hat)
        # clean, clean_amp = compute_cprs_stft(clean)
        # loss_amplitude = nn.functional.mse_loss(x_hat_amp, clean_amp)
        # loss_phase = nn.functional.mse_loss(x_hat, clean)
        # loss = 0.5 * loss_amplitude + 0.5 * loss_phase
        # return loss

        # x_hat, clean = x_hat.squeeze(1), clean.squeeze(1)
        # # return nn.functional.mse_loss(x_hat, clean)
        # x_hat_stft = torch.stft(x_hat, n_fft=512, win_length=25, hop_length=10, return_complex=True, pad_mode='constant')
        # clean_stft = torch.stft(clean, n_fft=512, win_length=25, hop_length=10, return_complex=True, pad_mode='constant')
        # # print('x_hat_stft.shape', x_hat_stft.shape, x_hat_stft.dtype)
        # x_hat_stft_as_real, clean_stft_as_real = torch.view_as_real(x_hat_stft), torch.view_as_real(clean_stft)
        # # print('x_hat_stft_as_real.shape', x_hat_stft_as_real.shape, x_hat_stft_as_real.dtype)
        # x_hat_stft_real, x_hat_stft_imag = x_hat_stft_as_real[:, :, :, 0], x_hat_stft_as_real[:, :, :, 1]
        # clean_stft_real, clean_stft_imag = clean_stft_as_real[:, :, :, 0], clean_stft_as_real[:, :, :, 1]
        # x_hat_amplitude = torch.sqrt(x_hat_stft_real**2 + x_hat_stft_imag**2)
        # clean_amplitude = torch.sqrt(clean_stft_real**2 + clean_stft_imag**2)
        # # print('x_hat_amplitude.shape', x_hat_amplitude.shape, x_hat_amplitude.dtype)
        # x_hat_amplitude_cpr = x_hat_amplitude ** 0.3
        # clean_amplitude_cpr = clean_amplitude ** 0.3
        # # print('x_hat_amplitude_cpr.shape', x_hat_amplitude_cpr.shape, x_hat_amplitude_cpr.dtype)
        # x_hat_cprs = x_hat_stft_as_real * (x_hat_amplitude_cpr / (x_hat_amplitude + 1e-12)).unsqueeze(3)
        # clean_cprs = clean_stft_as_real * (clean_amplitude_cpr / (clean_amplitude + 1e-12)).unsqueeze(3)
        # # print('x_hat_cprs.shape', x_hat_cprs.shape, x_hat_cprs.dtype)
        # loss_amplitude = nn.functional.mse_loss(x_hat_amplitude_cpr, clean_amplitude_cpr)
        # loss_phase = nn.functional.mse_loss(x_hat_cprs, clean_cprs)
        # loss = 0.5 * loss_amplitude + 0.5 * loss_phase
        # # return nn.functional.mse_loss(x_hat, clean)
        # return loss


    def training_step(self, batch: Tuple[List[Tensor], List[Tensor], List[Tensor]], batch_idx):
        x_hat, clean = self.process_batch(batch, batch_idx)
        loss = self.compute_loss(x_hat, clean)
        self.log("train_loss", loss, prog_bar=True, batch_size=24)
        if batch_idx == 0:
            self.logger.experiment.add_audio(f"{self.current_epoch}_{batch_idx}_x_hat", x_hat[0], sample_rate=16000)
            self.logger.experiment.add_audio(f"{self.current_epoch}_{batch_idx}_clean", clean[0], sample_rate=16000)
            # self.logger.experiment.add_video(f"{self.current_epoch}_{batch_idx}_video", batch[0][0].unsqueeze(0), fps=25)
        return loss

    def validation_step(self, batch: Tuple[List[Tensor], List[Tensor], List[Tensor]], batch_idx):
        x_hat, clean = self.process_batch(batch, batch_idx)
        loss = self.compute_loss(x_hat, clean)
        self.log("validation_loss", loss, prog_bar=True, sync_dist=True, batch_size=24)
        return loss

    def test_step(self, batch: Tuple[List[Tensor], List[Tensor], List[Tensor]], batch_idx):
        x_hat, clean = self.process_batch(batch, batch_idx)

        pesq = perceptual_evaluation_speech_quality(x_hat, clean, 16000, 'wb')
        sdr = signal_distortion_ratio(x_hat, clean)
        sisdr = scale_invariant_signal_distortion_ratio(x_hat, clean)
        loss = self.compute_loss(x_hat, clean)

        log = {
            "loss": loss,
            "pesq": pesq.mean(),
            "sdr": sdr.mean(),
            "sisdr": sisdr.mean()
        }

        self.log_dict(log, prog_bar=True, batch_size=24)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)
        return [optimizer], [scheduler]
    

# def loss(x_hat: Tensor, clean: Tensor):
#     x_hat, clean = x_hat.squeeze(1), clean.squeeze(1)
#     # return nn.functional.mse_loss(x_hat, clean)

#     x_hat_stft = torch.stft(x_hat, n_fft=512, win_length=400, hop_length=100, return_complex=True, pad_mode='constant')
#     clean_stft = torch.stft(clean, n_fft=512, win_length=400, hop_length=100, return_complex=True, pad_mode='constant')

#     # print('x_hat_stft.shape', x_hat_stft.shape, x_hat_stft.dtype)


#     x_hat_stft_as_real, clean_stft_as_real = torch.view_as_real(x_hat_stft), torch.view_as_real(clean_stft)

#     # print('x_hat_stft_as_real.shape', x_hat_stft_as_real.shape, x_hat_stft_as_real.dtype)

#     x_hat_stft_real, x_hat_stft_imag = x_hat_stft_as_real[:, :, :, 0], x_hat_stft_as_real[:, :, :, 1]
#     clean_stft_real, clean_stft_imag = clean_stft_as_real[:, :, :, 0], clean_stft_as_real[:, :, :, 1]
    

#     x_hat_amplitude = torch.sqrt(x_hat_stft_real**2 + x_hat_stft_imag**2)
#     clean_amplitude = torch.sqrt(clean_stft_real**2 + clean_stft_imag**2)

#     # print('x_hat_amplitude.shape', x_hat_amplitude.shape, x_hat_amplitude.dtype)


#     x_hat_amplitude_cpr = x_hat_amplitude ** 0.3
#     clean_amplitude_cpr = clean_amplitude ** 0.3

#     # print('x_hat_amplitude_cpr.shape', x_hat_amplitude_cpr.shape, x_hat_amplitude_cpr.dtype)


#     x_hat_cprs = x_hat_stft_as_real * (x_hat_amplitude_cpr / (x_hat_amplitude + 1e-12)).unsqueeze(3)
#     clean_cprs = clean_stft_as_real * (clean_amplitude_cpr / (clean_amplitude + 1e-12)).unsqueeze(3)

#     # print('x_hat_cprs.shape', x_hat_cprs.shape, x_hat_cprs.dtype)
   
#     loss_amplitude = nn.functional.mse_loss(x_hat_amplitude_cpr, clean_amplitude_cpr)
#     loss_phase = nn.functional.mse_loss(x_hat_cprs, clean_cprs)
#     loss = 0.5 * loss_amplitude + 0.5 * loss_phase

#     return loss

# def replace_denormals(x: torch.tensor, threshold=1e-8):
#     y = x.clone()
#     y[(x < threshold) & (x > -1.0 * threshold)] = threshold
#     return y

# def compute_cprs_stft(x: Tensor):
#     stft = torch.stft(x.squeeze(1), 512, return_complex=True)
#     # print('stft.abs', stft.abs().shape)
#     # print('stft.angle', stft.angle().shape)
#     stft_amp_cpr = stft.abs() ** 0.3
#     stft_phase = torch.exp(1j * stft.angle())

#     angle = replace_denormals(stft.angle())
#     mag = replace_denormals(stft.abs()**(1/1))
#     stft_cpr =  mag * torch.exp(1j * angle)

#     # stft_amp_cpr = replace_denormals(stft.abs() ** 0.3)
#     # stft_phase = replace_denormals(torch.exp(1j * stft.angle()))
#     # stft_cpr = stft * stft_amp_cpr * stft_phase

#     return torch.view_as_real(stft_cpr), mag

#     x = x.squeeze(1)
#     stft = torch.stft(x, 512, return_complex=True)
#     stft_real = torch.view_as_real(stft)
#     stft_abs = torch.abs(stft)
#     print(stft.shape)
#     print(stft_abs.shape)
#     print(torch.angle(stft).shape)
#     print(nn.functional.mse_loss(stft_abs, stft_abs))
#     print(nn.functional.mse_loss(stft_real, stft_real))
#     # stft = torch.view_as_real(torch.stft(x, 512, return_complex=True))
#     # x_real, x_imag = stft[0], stft[1]
#     # amp = torch.abs()

# def compute_loss(x_hat: Tensor, clean: Tensor) -> Tensor:
#     x_hat, x_hat_amp = compute_cprs_stft(x_hat)
#     clean, clean_amp = compute_cprs_stft(clean)
#     loss_amplitude = nn.functional.mse_loss(x_hat_amp, clean_amp)
#     loss_phase = nn.functional.mse_loss(x_hat, clean)
#     loss = 0.5 * loss_amplitude + 0.5 * loss_phase
#     return loss



if __name__ == "__main__":
    data = [(torch.rand((16, 25, 3, 96, 96)), torch.rand((16, 1, 16000)))]
    summary(AVE3Net(), input_data=data, col_names=["input_size",
                                                   "output_size",
                                                   "num_params"], depth=3)
    # summary(AVE3Net(), input_size=(1, 60000))

    # x_hat, clean = torch.rand((24, 1, 32000)), torch.rand((24, 1, 32000))

    # datamodule = DataModule(batch_size=24)
    # datamodule.setup("fit")

    # train_dataloader = datamodule.train_dataloader()
    # i = iter(train_dataloader)
    # batch: Tuple[Tensor, Tensor, Tensor] = next(i)
    # vframes, noisy, clean = batch

    # s, sa = compute_cprs_stft(clean)

    # print(sa.shape, sa.dtype)
    # print(s.shape, s.dtype)

    # loss = compute_loss(noisy, clean)
    # print('loss', loss)
