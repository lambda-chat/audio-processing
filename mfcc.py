import librosa
import os

work_dir = "work/audio"
filename_in = "01_BASS_A.wav"
filepath_in = os.path.join(work_dir, filename_in)

y, sampling_rate = librosa.load(filepath_in)
mfcc = librosa.feature.mfcc(y, sr=sampling_rate)

from librosa import display
from matplotlib import pyplot as plt
import numpy as np

# display.specshow(mfcc, sr=sampling_rate, x_axis="time")
# plt.colorbar()
# plt.show()

# print(mfcc.shape) # (20, 1352)

S = librosa.feature.melspectrogram(y, sr=sampling_rate, n_mels=128)
log_S = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(12, 4))
librosa.display.specshow(log_S, sr=sampling_rate, x_axis="time", y_axis="mel")
plt.title("mel power spectrogram")
plt.colorbar(format="%02.0f dB")
plt.tight_layout()
plt.show()