from pydub import AudioSegment
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
import os
import math

work_dir = "work/audio"
filename_in = "01_BASS_A_mono.wav"
filepath_in = os.path.join(work_dir, filename_in)
sound = AudioSegment.from_file(filepath_in, "wav")

print("sample_width = {}".format(sound.sample_width))
print("frame_rate = {}".format(sound.frame_rate))
print("channels = {}".format(sound.channels))

# 本体のデータ
ys_mono = np.array(sound.get_array_of_samples())

# LP, HP 用それぞれコピーを用意
ys_lp = ys_mono
ys_hp = ys_mono

# FFT したもの（周波数順にソートされているわけではないことに注意）
ws_lp = fftpack.fft(ys_lp)
ws_hp = fftpack.fft(ys_hp)

# 周波数（周波数順にソートされているわけではないことに注意）
freq = fftpack.fftfreq(n = len(ys_mono), d = 1.0 / sound.frame_rate)

# low pass gain
def lin_lp_gain(freq, begin = 20, end = 10000, amount = -42):
    freq = np.abs(freq)
    if freq <= begin:
        return 0
    elif freq >= end:
        return amount
    else:
        return amount * math.log(end / freq, end / begin)

# high pass gain
def lin_hp_gain(freq, begin = 20, end = 10000, amount = -42):
    freq = np.abs(freq)
    if freq <= begin:
        return amount
    elif freq >= end:
        return 0
    else:
        return amount * math.log(freq / begin, end - begin)

# LR の音量の調整
for i in range(len(freq)):
    fq = freq[i]
    ws_lp[i] *= 10 ** (lin_lp_gain(fq) / 20)
    ws_hp[i] *= 10 ** (lin_hp_gain(fq) / 20)

# 逆フーリエ変換
ys_lp = fftpack.ifft(ws_lp)
ys_hp = fftpack.ifft(ws_hp)

# HP の位相反転
ys_hp = -ys_hp

# LR 供用の出力
ys_fx = ys_lp + ys_hp

# 出力用に整形 
data_fx = np.stack([ys_fx.astype('int16'), ys_fx.astype('int16')]).flatten('F').tobytes()
fx_sound = AudioSegment(
    data = data_fx,
    sample_width=sound.sample_width,
    frame_rate=sound.frame_rate,
    channels=2)

# 一旦出力
if True:
    filename_out = "01_BASS_A_mono2fx.wav"
    filepath_out = os.path.join(work_dir, filename_out)
    fx_sound.export(filepath_out, "wav", bitrate="192k")

# ys_fx を 20 ms 後ろにずらす（前半 20 ms は無音、後半 20 ms は除去）
ys_fx_latency = np.pad(ys_fx, (int(sound.frame_rate * 20 / 1000), 0), 'constant')[:len(ys_fx)]

# ys_fx を 5 dB 大きくする
ys_fx_latency *= 10 ** (5 / 20)

# 左右に設定
ys_fx_L = ys_fx_latency
ys_fx_R = -ys_fx_latency

# 元のモノラルと足し上げる
ys_sur_L = ys_mono + ys_fx_L
ys_sur_R = ys_mono + ys_fx_R

# 出力用に整形 
data_sur = np.stack([ys_sur_L.astype('int16'), ys_sur_R.astype('int16')]).flatten('F').tobytes()
fx_sound = AudioSegment(
    data = data_sur,
    sample_width=sound.sample_width,
    frame_rate=sound.frame_rate,
    channels=2)

if True:
    filename_out = "01_BASS_A_surround.wav"
    filepath_out = os.path.join(work_dir, filename_out)
    fx_sound.export(filepath_out, "wav", bitrate="192k")