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

# L, R それぞれコピーを用意
ys_left = ys_mono
ys_right = ys_mono

# L, R を FFT したもの（周波数順にソートされているわけではないことに注意）
ws_left = fftpack.fft(ys_left)
ws_right = fftpack.fft(ys_right)

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
    left_adj = lin_lp_gain(fq)
    right_adj = lin_hp_gain(fq)
    ws_left[i] *= 10 ** (left_adj / 20)
    ws_right[i] *= 10 ** (right_adj / 20)

# 逆フーリエ変換
ys_left = fftpack.ifft(ws_left).astype('int16')
ys_right = fftpack.ifft(ws_right).astype('int16')

# right の位相反転
ys_right = -ys_right

ys_stereo = np.stack([ys_left, ys_right]).flatten('F')

fx_sound = AudioSegment(
    data = ys_stereo.astype("int16").tobytes(),
    sample_width=sound.sample_width,
    frame_rate=sound.frame_rate,
    channels=2)

filename_out = "01_BASS_A_mono2fx.wav"
filepath_out = os.path.join(work_dir, filename_out)
fx_sound.export(filepath_out, "wav", bitrate="192k")