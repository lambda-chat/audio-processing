from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import os

work_dir = "work/audio"
filename_in = "01_BASS_A.mp3"
filepath_in = os.path.join(work_dir, filename_in)
sound = AudioSegment.from_file(filepath_in, "mp3")
sound = sound.set_channels(1)

filename_out = "01_BASS_A_mono.wav"
filepath_out = os.path.join(work_dir, filename_out)
sound.export(filepath_out, format="wav", bitrate="192k")