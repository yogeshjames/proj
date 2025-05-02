import os
import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

# ──────────────────────────────────────────────────────────────────────────────
# 1) Dataset: matches noisy→clean by suffix, resamples, pads/truncates
# ──────────────────────────────────────────────────────────────────────────────
class SpeechDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, sample_rate=16000, max_len=16000):
        self.sample_rate = sample_rate
        self.max_len     = max_len
        noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
        clean_paths = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
        clean_map   = {os.path.basename(p): p for p in clean_paths}
        pairs = []
        for npath in noisy_paths:
            clean_name = os.path.basename(npath).split('_')[-1]
            if clean_name in clean_map:
                pairs.append((npath, clean_map[clean_name]))
        if not pairs:
            raise RuntimeError(f"No matching files in {noisy_dir} vs {clean_dir}")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        noisy, sr1 = torchaudio.load(noisy_path)   # [C, T]
        clean, sr2 = torchaudio.load(clean_path)
        if sr1 != self.sample_rate:
            noisy = F.resample(noisy, orig_freq=sr1, new_freq=self.sample_rate)
        if sr2 != self.sample_rate:
            clean = F.resample(clean, orig_freq=sr2, new_freq=self.sample_rate)
        noisy, clean = noisy[0], clean[0]
        if noisy.numel() < self.max_len:
            pad = self.max_len - noisy.numel()
            noisy = torch.cat([noisy, torch.zeros(pad)])
            clean = torch.cat([clean, torch.zeros(pad)])
        else:
            noisy = noisy[:self.max_len]
            clean = clean[:self.max_len]
        return noisy.unsqueeze(1), clean.unsqueeze(1)  # [T,1]

# ──────────────────────────────────────────────────────────────────────────────
# 2) RNN filter model
# ──────────────────────────────────────────────────────────────────────────────
class RNNFilter(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Training + FIR & RNN metrics, save model and plot
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RNN denoiser with FIR comparison and graphs")
    parser.add_argument("--noisy_dir",  required=True)
    parser.add_argument("--clean_dir",  required=True)
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--sr",         type=int,   default=16000)
    parser.add_argument("--max_len",    type=int,   default=16000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds     = SpeechDataset(args.noisy_dir, args.clean_dir, sample_rate=args.sr, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model     = RNNFilter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Precompute FIR coefficients
    fir_coeff = firwin(101, cutoff=0.1)

    # Lists to store per-epoch metrics
    mse_fir_list = []
    mse_rnn_list = []
    snr_fir_list = []
    snr_rnn_list = []

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            out = model(noisy)
            loss = criterion(out, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_rnn_loss = total_loss / len(loader)

        # Evaluate FIR and RNN on first sample
        noisy_s, clean_s = ds[0]
        noisy_np = noisy_s.squeeze(1).numpy()
        clean_np = clean_s.squeeze(1).numpy()
        with torch.no_grad():
            rnn_out = model(noisy_s.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        fir_out = lfilter(fir_coeff, 1.0, noisy_np)

        mse_fir = np.mean((clean_np - fir_out)**2)
        mse_rnn = np.mean((clean_np - rnn_out)**2)
        snr_fir = 10*np.log10(np.mean(clean_np**2)/mse_fir)
        snr_rnn = 10*np.log10(np.mean(clean_np**2)/mse_rnn)

        mse_fir_list.append(mse_fir)
        mse_rnn_list.append(mse_rnn)
        snr_fir_list.append(snr_fir)
        snr_rnn_list.append(snr_rnn)

        print(f"[Epoch {epoch}/{args.epochs}]  "
              f"MSE_FIR={mse_fir:.6f}, MSE_RNN={mse_rnn:.6f}  |  "
              f"SNR_FIR={snr_fir:.2f} dB, SNR_RNN={snr_rnn:.2f} dB")

    # Save the trained model
    torch.save(model.state_dict(), "rnn_filter.pth")
    print("Training complete. Model saved to rnn_filter.pth")

    # Plot metrics over epochs
    epochs = range(1, args.epochs+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, mse_fir_list, label='MSE_FIR')
    plt.plot(epochs, mse_rnn_list, label='MSE_RNN')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title('MSE per Epoch'); plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(epochs, snr_fir_list, label='SNR_FIR')
    plt.plot(epochs, snr_rnn_list, label='SNR_RNN')
    plt.xlabel('Epoch'); plt.ylabel('SNR (dB)'); plt.title('SNR per Epoch'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
