# proj

Noisy data dir

--noisy_dir

Path to directory containing noisy .wav files

required

Clean data dir

--clean_dir

Path to directory containing clean .wav files

required

Number of epochs

--epochs

How many full passes over the training set

20

Batch size

--batch_size

Number of samples per training batch

8

Learning rate

--lr

Initial learning rate for the Adam optimizer

1e-3

Sample rate

--sr

Target sampling rate (Hz); will resample if different

16000

Sequence length

--max_len

Maximum number of audio samples per input sequence

16000

Usage

Run the training and plotting in a single command. Example:    
python k.py --noisy_dir MS-SNSD/NoisySpeech_training --clean_dir MS-SNSD/CleanSpeech_training --epochs 20 --batch_size 8 --lr 1e-3 --sr 16000 --max_len 16000


after this run 
python l.py --metrics_file metrics.npz --model_path rnn_filter.pth --noisy_dir MS-SNSD/NoisySpeech_training --clean_dir MS-SNSD/CleanSpeech_training --sr 16000 --max_len 16000
 to get the grplagh outputs 
