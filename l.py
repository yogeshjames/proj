# visualize_results.py

import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
from scipy.signal import firwin, lfilter

# 1) RNN definition (must match your training)
class RNNFilter(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)

def load_sample(noisy_dir, clean_dir, sr, max_len, idx=0):
    noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    clean_map   = {os.path.basename(p):p for p in glob.glob(os.path.join(clean_dir,"*.wav"))}
    npath = noisy_paths[idx]
    cname = os.path.basename(npath).split('_')[-1]
    cpath = clean_map[cname]
    noisy, s1 = torchaudio.load(npath)
    clean, s2 = torchaudio.load(cpath)
    if s1!=sr: noisy=F.resample(noisy,s1,sr)
    if s2!=sr: clean=F.resample(clean,s2,sr)
    noisy = noisy[0].numpy(); clean=clean[0].numpy()
    if noisy.shape[0]<max_len:
        pad=max_len-noisy.shape[0]
        noisy=np.pad(noisy,(0,pad)); clean=np.pad(clean,(0,pad))
    else:
        noisy=noisy[:max_len]; clean=clean[:max_len]
    return noisy, clean

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_file", required=True)
    p.add_argument("--model_path",   required=True)
    p.add_argument("--noisy_dir",    required=True)
    p.add_argument("--clean_dir",    required=True)
    p.add_argument("--sr",      type=int, default=16000)
    p.add_argument("--max_len", type=int, default=16000)
    args = p.parse_args()

    # ----- 1) Plot MSE & SNR vs. Epoch -----
    data = np.load(args.metrics_file)
    epochs = np.arange(1, len(data['mse_fir'])+1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, data['mse_fir'], label='MSE_FIR')
    plt.plot(epochs, data['mse_rnn'], label='MSE_RNN')
    plt.title('MSE vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, data['snr_fir'], label='SNR_FIR')
    plt.plot(epochs, data['snr_rnn'], label='SNR_RNN')
    plt.title('SNR vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('SNR (dB)')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig("epoch_metrics.png")
    print("✓ Saved epoch_metrics.png")
    plt.show()

    # ----- 2) Waveform comparison -----
    noisy, clean = load_sample(args.noisy_dir, args.clean_dir, args.sr, args.max_len)
    # load model
    model = RNNFilter().cpu()
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(-1)
        rnn_out = model(inp).squeeze().numpy()

    fir_coeff = firwin(101, cutoff=0.1)
    fir_out = lfilter(fir_coeff, 1.0, noisy)

    mse_fir = np.mean((clean - fir_out)**2)
    mse_rnn = np.mean((clean - rnn_out)**2)
    snr_fir = 10*np.log10(np.mean(clean**2)/mse_fir)
    snr_rnn = 10*np.log10(np.mean(clean**2)/mse_rnn)

    t = np.arange(args.max_len) / args.sr
    plt.figure(figsize=(10,5))
    plt.plot(t, clean, '--', label='Clean')
    plt.plot(t, noisy, alpha=0.3, label='Noisy')
    plt.plot(t, fir_out,     label=f'FIR (SNR {snr_fir:.2f} dB)')
    plt.plot(t, rnn_out,     ':', label=f'RNN (SNR {snr_rnn:.2f} dB)')
    plt.title('Waveform Comparison')
    plt.xlabel('Time (s)'); plt.ylabel('Amplitude')
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig("waveform_comparison.png")
    print("✓ Saved waveform_comparison.png")
    plt.show()

if __name__=="__main__":
    main()
