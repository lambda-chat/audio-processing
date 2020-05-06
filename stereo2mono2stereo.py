from pydub import AudioSegment
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import math
import os

src_dir = "E/MUSIC"
out_dir = "work/audio"

def get_audio_segment(filename: str, format: str) -> AudioSegment:
    filepath = os.path.join(src_dir, filename)
    return AudioSegment.from_file(filepath, format)

def monoralize(sound: AudioSegment) -> AudioSegment:
    return sound.set_channels(1)

def monoral_to_stereo(sound: AudioSegment) -> AudioSegment:
    # 各種情報の出力（確認用）
    print("sample_width = {}".format(sound.sample_width))
    print("frame_rate = {}".format(sound.frame_rate))
    print("channels = {}".format(sound.channels))
    assert(sound.channels == 1)

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

    # LP, HP の音量の調整
    for i in range(len(freq)):
        fq = freq[i]
        ws_lp[i] *= 10 ** (lin_lp_gain(fq) / 20)
        ws_hp[i] *= 10 ** (lin_hp_gain(fq) / 20)

    # 逆フーリエ変換
    ys_lp = fftpack.ifft(ws_lp)
    ys_hp = fftpack.ifft(ws_hp)

    # HP の位相反転
    ys_hp = -ys_hp

    # LR 供用の出力（Low: 変化せず → Mid: 打ち消される → High: 位相反転）
    ys_fx = ys_lp + ys_hp

    # ys_fx を 20 ms 後ろにずらす（前半 20 ms は無音、後半 20 ms は除去）
    ys_fx_latency = np.pad(ys_fx, (int(sound.frame_rate * 20 / 1000), 0), 'constant')[:len(ys_fx)]

    # ys_fx を 5 dB 大きくする
    ys_fx_latency *= 10 ** (5 / 20)

    # 左右に設定
    ys_fx_L = ys_fx_latency
    ys_fx_R = -ys_fx_latency # L, R 片方のみ反転すること

    # 元のモノラルと足し上げる
    ys_sur_L = ys_mono + ys_fx_L
    ys_sur_R = ys_mono + ys_fx_R

    # 出力用に整形
    if sound.sample_width == 1:
        stype = 'int8'
    elif sound.sample_width == 2:
        stype = 'int16'
    elif sound.sample_width == 4:
        stype = 'int32'
    else:
        raise Exception("available sample_width is 1 or 2 or 4 as for now")

    data_sur = np.stack([ys_sur_L.astype(stype), ys_sur_R.astype(stype)]).flatten('F').tobytes()
    sound = AudioSegment(
        data = data_sur,
        sample_width=sound.sample_width,
        frame_rate=sound.frame_rate,
        channels=2)

    # 終わり
    return sound

def output_audio_segment(sound: AudioSegment, filename: str, format: str):
    filepath = os.path.join(out_dir, filename)
    sound.export(filepath, format, bitrate="192k")

def main():
    filename_in = "000826_0039.wav"
    filename_out = "processed_" + filename_in

    # 処理
    sound = get_audio_segment(filename_in, "wav")
    sound = monoralize(sound)
    sound = monoral_to_stereo(sound)
    output_audio_segment(sound, filename_out, "wav")

if __name__ == "__main__":
    main()