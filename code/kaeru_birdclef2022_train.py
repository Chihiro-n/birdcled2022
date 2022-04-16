import cv2
import audioread
import logging
import os
import sys
#sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
sys.path.append('/content/drive/MyDrive/python/kaggle/birdclef-2022/input/pytorch-image-models/pytorch-image-models-master')
import random
import time
import warnings

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from contextlib import contextmanager
from pathlib import Path
from typing import List
from typing import Optional
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GroupKFold

from albumentations.core.transforms_interface import ImageOnlyTransform
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm

import albumentations as A
import albumentations.pytorch.transforms as T



IMAGE_PATH = '../input/birdclef2022-audio-image-dataset/'

train = pd.read_csv(IMAGE_PATH + 'train_folds.csv')
train["file_path"] = IMAGE_PATH + train['filename'] + '.npy'

print(train.shape)
train.head()


class CFG:
    EXP_ID = 'N001' 

    ######################
    # Globals #
    ######################
    seed = 42
    epochs = 35 ## chg 5 to 35
    train = True
    folds = [0,1,2,3,4]
    img_size = 128
    main_metric = "epoch_f1_at_03"
    minimize_metric = False

    ######################
    # Data #
    ######################
    train_datadir = Path("../input/birdclef-2022/train_audio")
    train_csv = "../input/birdclef-2022/train_metadata.csv"

    ######################
    # Dataset #
    ######################
    transforms = {
        "train": [{"name": "Normalize"}],
        "valid": [{"name": "Normalize"}]
    }
    period = 5
    n_mels = 224
    fmin = 20
    fmax = 16000
    n_fft = 2048
    hop_length = 512
    sample_rate = 32000
    melspectrogram_parameters = {
        "n_mels": 224,
        "fmin": 20,
        "fmax": 16000
    }

    target_columns = 'afrsil1 akekee akepa1 akiapo akikik amewig aniani apapan arcter \
                      barpet bcnher belkin1 bkbplo bknsti bkwpet blkfra blknod bongul \
                      brant brnboo brnnod brnowl brtcur bubsan buffle bulpet burpar buwtea \
                      cacgoo1 calqua cangoo canvas caster1 categr chbsan chemun chukar cintea \
                      comgal1 commyn compea comsan comwax coopet crehon dunlin elepai ercfra eurwig \
                      fragul gadwal gamqua glwgul gnwtea golphe grbher3 grefri gresca gryfra gwfgoo \
                      hawama hawcoo hawcre hawgoo hawhaw hawpet1 hoomer houfin houspa hudgod iiwi incter1 \
                      jabwar japqua kalphe kauama laugul layalb lcspet leasan leater1 lessca lesyel lobdow lotjae \
                      madpet magpet1 mallar3 masboo mauala maupar merlin mitpar moudov norcar norhar2 normoc norpin \
                      norsho nutman oahama omao osprey pagplo palila parjae pecsan peflov perfal pibgre pomjae puaioh \
                      reccar redava redjun redpha1 refboo rempar rettro ribgul rinduc rinphe rocpig rorpar rudtur ruff \
                      saffin sander semplo sheowl shtsan skylar snogoo sooshe sooter1 sopsku1 sora spodov sposan \
                      towsol wantat1 warwhe1 wesmea wessan wetshe whfibi whiter whttro wiltur yebcar yefcan zebdov'.split()

    ######################
    # Loaders #
    ######################
    loader_params = {
        "train": {
            "batch_size": 16, 
            "num_workers": 0,
            "shuffle": True
        },
        "valid": {
            "batch_size": 32,
            "num_workers": 0,
            "shuffle": False
        }
    }

    ######################
    # Split #
    ######################
    split = "StratifiedKFold"
    split_params = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42
    }

    ######################
    # Model #
    ######################
    base_model_name = "tf_efficientnet_b0_ns"
    pooling = "max"
    pretrained = True
    num_classes = 152
    in_channels = 3

    N_FOLDS = 5
    LR = 1e-3
    T_max=10
    min_lr=1e-6

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def get_transforms(phase: str):
    transforms = CFG.transforms
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else \
                trns_conf["params"]
            if globals().get(trns_name) is not None:
                trns_cls = globals()[trns_name]
                trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return Compose(trns_list)
        else:
            return None
        
        
class Normalize:
    def __call__(self, y: np.ndarray):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


# Mostly taken from https://www.kaggle.com/hidehisaarai1213/rfcx-audio-data-augmentation-japanese-english
class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear").squeeze(1)

    return output



class AttBlockV2(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)


class TimmSED(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False, num_classes=24, in_channels=1):
        super().__init__()

        self.spec_augmenter = SpecAugmentation(time_drop_width=64//2, time_stripes_num=2,
                                               freq_drop_width=8//2, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(CFG.n_mels)

        base_model = timm.create_model(
            base_model_name, pretrained=pretrained, in_chans=in_channels)
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        if hasattr(base_model, "fc"):
            in_features = base_model.fc.in_features
        else:
            in_features = base_model.classifier.in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block = AttBlockV2(
            in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        

    def forward(self, input_data):
        x = input_data # (batch_size, 3, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            if random.random() < 0.25:
                x = self.spec_augmenter(x)

        x = x.transpose(2, 3)

        x = self.encoder(x)
        
        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        logit = torch.sum(norm_att * self.att_block.cla(x), dim=2)
        segmentwise_logit = self.att_block.cla(x).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output,
            'logit': logit,
            'framewise_logit': framewise_logit,
        }

        return output_dict

# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
            (1. - probas)**self.gamma * bce_loss + \
            (1. - targets) * probas**self.gamma * bce_loss
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


# ====================================================
# Training helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        # self.y_pred.extend(torch.sigmoid(y_pred).cpu().detach().numpy().tolist())
        # self.y_pred.extend(y_pred["clipwise_output"].max(axis=1)[0].cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred["clipwise_output"].cpu().detach().numpy().tolist())

    @property
    def avg(self):
        self.f1_005 = metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.05, average="micro")
        self.f1_01 = metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.1, average="micro")
        self.f1_03 = metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.3, average="micro")
        self.f1_05 = metrics.f1_score(np.array(self.y_true), np.array(self.y_pred) > 0.5, average="micro")
        
        return {
            "f1_at_005" : self.f1_005,
            "f1_at_01" : self.f1_01,
            "f1_at_03" : self.f1_03,
            "f1_at_05" : self.f1_05,
        }


def loss_fn(logits, targets):
    loss_fct = BCEFocal2WayLoss()
    loss = loss_fct(logits, targets)
    return loss

        
def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for data in tk0:
        optimizer.zero_grad()
        inputs = data['image'].to(device)
        targets = data['primary_targets'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    valid_preds = []
    with torch.no_grad():
        for data in tk0:
            inputs = data['image'].to(device)
            targets = data['primary_targets'].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg

mean = (0.485, 0.456, 0.406) # RGB
std = (0.229, 0.224, 0.225) # RGB

albu_transforms = {
    'train' : A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.Cutout(max_h_size=5, max_w_size=16),
                A.CoarseDropout(max_holes=4),
            ], p=0.5),
            A.Normalize(mean, std),
    ]),
    'valid' : A.Compose([
            A.Normalize(mean, std),
    ]),
}


class WaveformDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 mode='train'):
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        
        wav_path = sample["file_path"]
        labels = sample["primary_label"]
        
        image = np.load(wav_path) # (224, 313, 3)
        image = albu_transforms[self.mode](image=image)['image']
        image = image.T

        targets = np.zeros(len(CFG.target_columns), dtype=float)
        for ebird_code in labels.split():
            targets[CFG.target_columns.index(ebird_code)] = 1.0

        return {
            "image": image,
            "primary_targets": targets,
        }


OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings("ignore")
logger = init_logger(log_file=f"train_{CFG.EXP_ID}.log")

# environment
set_seed(CFG.seed)
device = get_device()


# main loop
for fold in range(5):
    if fold not in CFG.folds:
        continue
    logger.info("=" * 90)
    logger.info(f"Fold {fold} Training")
    logger.info("=" * 90)

    trn_df = train[train['kfold']!=fold].reset_index(drop=True)
    val_df = train[train['kfold']==fold].reset_index(drop=True)

    loaders = {
        phase: torchdata.DataLoader(
            WaveformDataset(
                df_,
                mode=phase
            ),
            **CFG.loader_params[phase])  # type: ignore
        for phase, df_ in zip(["train", "valid"], [trn_df, val_df])
    }

    model = TimmSED(
        base_model_name=CFG.base_model_name,
        pretrained=CFG.pretrained,
        num_classes=CFG.num_classes,
        in_channels=CFG.in_channels)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)

    model = model.to(device)


    p = 0
    min_loss = 999
    best_score = -np.inf

    for epoch in range(CFG.epochs):

        logger.info("Starting {} epoch...".format(epoch+1))

        start_time = time.time()

        train_avg, train_loss = train_fn(model, loaders['train'], device, optimizer, scheduler)

        valid_avg, valid_loss = valid_fn(model, loaders['valid'], device)
        scheduler.step()
        
        elapsed = time.time() - start_time
        
        logger.info(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        logger.info(f"Epoch {epoch+1} - train_f1_at_005:{train_avg['f1_at_005']:0.5f}  valid_f1_at_005:{valid_avg['f1_at_005']:0.5f}")
        logger.info(f"Epoch {epoch+1} - train_f1_at_01:{train_avg['f1_at_01']:0.5f}  valid_f1_at_01:{valid_avg['f1_at_01']:0.5f}")
        logger.info(f"Epoch {epoch+1} - train_f1_at_03:{train_avg['f1_at_03']:0.5f}  valid_f1_at_03:{valid_avg['f1_at_03']:0.5f}")
        logger.info(f"Epoch {epoch+1} - train_f1_at_05:{train_avg['f1_at_05']:0.5f}  valid_f1_at_05:{valid_avg['f1_at_05']:0.5f}")

        if valid_avg['f1_at_03'] > best_score:
            logger.info(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['f1_at_03']}")
            logger.info(f"other scores here... {valid_avg['f1_at_03']}, {valid_avg['f1_at_05']}")
            torch.save(model.state_dict(), f'fold-{fold}.bin')
            best_score = valid_avg['f1_at_03']
