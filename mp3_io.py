from pydub import AudioSegment
import librosa
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
import os

work_dir = "work/audio"
filename_in = "01_BASS_A.mp3"
filepath_in = os.path.join(work_dir, filename_in)
sound = AudioSegment.from_file(filepath_in, "mp3")

samples = np.array(sound.get_array_of_samples())
assert(sound.channels == 2)

left = samples[0::sound.channels]
right = samples[1::sound.channels]

filename_out = "01_BASS_A.wav"
filepath_out = os.path.join(work_dir, filename_out)
sound.export(filepath_out, "wav", bitrate="192k")