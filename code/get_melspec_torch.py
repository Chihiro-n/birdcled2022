import os
import random
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import model_selection
from sklearn import preprocessing
from  sklearn.model_selection  import StratifiedKFold
import IPython.display as ipd

import IPython.display
from IPython.display import display

import requests
import matplotlib.pyplot as plt

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import soundfile as sf
from  soundfile import SoundFile
import numpy as np

from tqdm.notebook import tqdm
from pathlib import Path
import joblib, json


def get_audio_info(filepath):
    """Get some properties from  an audio file"""
    with SoundFile(filepath) as f:
        sr = f.samplerate
        frames = f.frames
        duration = float(frames)/sr
    return {"frames": frames, "sr": sr, "duration": duration}


def make_df(n_splits=5, seed=42, nrows=None):
    
    df = pd.read_csv(DATA_ROOT/"train_metadata.csv", nrows=nrows)

    LABEL_IDS = {label: label_id for label_id,label in enumerate(sorted(df["primary_label"].unique()))}
    
    #df = df.iloc[PART_INDEXES[PART_ID]: PART_INDEXES[PART_ID+1]]

    df["label_id"] = df["primary_label"].map(LABEL_IDS)

    df["filepath"] = [str(TRAIN_AUDIO_ROOT/filename) for filename in df["filename"].values ]

    pool = joblib.Parallel(4)
    mapper = joblib.delayed(get_audio_info)
    tasks = [mapper(filepath) for filepath in df.filepath]

    df = pd.concat([df, pd.DataFrame(pool(tqdm(tasks)))], axis=1, sort=False)
    
    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    splits = skf.split(np.arange(len(df)), y=df.label_id.values)
    df["fold"] = -1

    for fold, (train_set, val_set) in enumerate(splits):
        
        df.loc[df.index[val_set], "fold"] = fold

    return LABEL_IDS, df


def get_spectrogram(
    waveform,
    n_fft = 400,
    win_len = None,
    hop_len = None,
    power = 2.0,
):
    #waveform, _ = get_speech_sample()
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

SR = 32_000
DURATION = 7 
SEED = 666

sr=SR
n_mels=128
fmin=0
fmax=None
duration=DURATION

audio_length = duration*sr

## 0.666はかぶりね
step=int(DURATION*0.666*SR)

res_type="kaiser_fast"
resample=True

output_dir = "/content/output/"
if not os.path.exists(output_dir):
    os.mkdir( output_dir )

TRAIN_AUDIO_IMAGES_SAVE_ROOT = Path("/content/output")

class MelSpecComputer:
    def __init__(self, sr, device="cpu", n_fft = 400, win_len = None, hop_len = None, power = 2.0, n_mels=128, fmin=None, fmax=None, **kwargs):
        
        self.device=device
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = self.sr//10
        self.win_len =self.n_fft
        self.hop_len = hop_len
        self.fmin = fmin
        self.fmax = fmax
        self.power = 1.0
        kwargs["n_fft"] = kwargs.get("n_fft", self.sr//10)
        kwargs["hop_length"] = kwargs.get("hop_length", self.sr//(10*4))
        self.kwargs = kwargs

    def __call__(self, y):

        melspectrogram = T.MelSpectrogram(
            n_fft=self.n_fft,
            win_length=self.win_len,
            hop_length=self.sr//(10*4),
            center=True,
            #normalized = True,
            #norm="slaney",
            #pad_mode="reflect",
            power=self.power,
            n_mels=self.n_mels
        )

        if device == "cuda":
            melspectrogram = melspectrogram.cuda()
        
        y = melspectrogram(y)

        return y

def crop_or_pad(y, length, device, sr=SR, train=True, probs=None):
    if len(y) <= length:
        if device == "cuda":
           y = torch.cat([y, torch.zeros(length - len(y)).to('cuda') ])
        else:
           y = torch.cat([y, torch.zeros(length - len(y))])
    else:
        if not train:
            start = 0
        elif probs is None:
            start = np.random.randint(len(y) - length)
        else:
            start = (
                    np.random.choice(np.arange(len(probs)), p=probs) + np.random.random()
            )
            start = int(sr * (start))

        y = y[start: start + length]

    return y

def mono_to_color(X, device, eps=1e-6, mean=None, std=None):

    # Standardize
    X = torch.stack([X, X, X], dim=-1)

    mean =  X.mean()
    std = X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = torch.clip(X, _min, _max)
        V = 255 * ((V - _min) / (_max - _min))
        V = V.to(torch.uint8)
    else:
        if device == "cuda":
            V = torch.zeros_like(X, dtype=torch.uint8).to('cuda')
        else:
            V = torch.zeros_like(X, dtype=torch.uint8)
    
    return V


class AudioToImage:
    def __init__(self, device="cpu", sr=SR, n_mels=128, fmin=0, fmax=None, duration=DURATION, step=None, res_type="kaiser_fast", resample=True):

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr//2

        self.duration = duration
        self.audio_length = self.duration*self.sr
        self.step = step or self.audio_length
        
        self.res_type = res_type
        self.resample = resample

        self.device = device

        self.mel_spec_computer = MelSpecComputer(device=self.device, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin,
                                                 fmax=self.fmax)
        
    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio) 
        image = mono_to_color(melspec, self.device)
#         image = normalize(image, mean=None, std=None)
        return image

    def __call__(self, row ,save=True):
#       max_audio_duration = 10*self.duration
#       init_audio_length = max_audio_duration*row.sr
        
#       start = 0 if row.duration <  max_audio_duration else np.random.randint(row.frames - init_audio_length)
    
      #audio, orig_sr = sf.read(row.filepath, dtype="float32")
        audio, orig_sr = torchaudio.load(row.filepath, normalize=True)

        if self.device == "cuda":
            audio = audio.to('cuda')

        audio = torch.t(audio)  ### ajust librosa format
        audio = torch.mean(audio, 1)
    
        audios = [audio[i:i+audio_length] for i in range(0, max(1, len(audio) - audio_length + 1), step)]
        audios[-1] = crop_or_pad(audios[-1],audio_length, self.device)
        images = [self.audio_to_image(audio) for audio in audios]
        images = torch.stack(images)
            
        if save:
            path = TRAIN_AUDIO_IMAGES_SAVE_ROOT/f"{row.filename[:-4]}.pt"
            path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(images, str(path))
        else:
            return  row.filename, images

def get_audios_as_images(df, device):
    pool = joblib.Parallel(2)
    
    converter = AudioToImage(step=int(DURATION*0.666*SR), device=device)
    mapper = joblib.delayed(converter)
    tasks = [mapper(row) for row in df.itertuples(False)]
    
    pool(tqdm(tasks))


def main():
    SR = 32_000
    DURATION = 7 
    SEED = 42

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    DATA_ROOT = Path("../input/birdclef-2022")
    TRAIN_AUDIO_ROOT = Path("../input/birdclef-2022/train_audio")
    
    MYDRIVE = Path("/content/drive/MyDrive/python/kaggle/birdclef-2022/input")

    data_dir = str(MYDRIVE/"df_metadata.csv")
    if not os.path.exists(data_dir):
        LABEL_IDS, df = make_df(nrows=None, SEED=SEED)
        df.to_csv(MYDRIVE/"df_metadata.csv",index=False)
        with open(MYDRIVE/"LABEL_IDS.json", "w") as f:
            json.dump(LABEL_IDS, f)
    else:
        df = pd.read_csv(data_dir)

    #_df = df[df["filename"]=="afrsil1/XC175522.ogg"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device -> ", device)

    """
    converter = AudioToImage(step=int(DURATION*0.666*SR), device=device)
    save = True

    for row in _df.itertuples():
        if save:
            converter(row,save=save)
        else:
            _,image =  converter(row,save=save)
        break
    """
    get_audios_as_images(df, device)

if __name__ == "__main__":
    main()
