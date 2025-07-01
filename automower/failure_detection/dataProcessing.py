#Script for splitting longer audio clips into smaller ones and saving them
import librosa
import soundfile
import os

label = "stuckBlade"

#Choose which file to split
inputFile = f"misc/husqvarnaData/{label}.wav"

#Output to correct label folder
outputDir = f"misc/husqvarnaDataSplit10s/{label}/"
os.makedirs(outputDir, exist_ok=True) 

inputAudio, sampleRate = librosa.load(inputFile, sr=None)

length = 10
size = int(length * sampleRate)

#Loop through the audio file in size segments
s = 0
for i in range(0, len(inputAudio), size):
    sample = inputAudio[i:i+size]
    soundfile.write(outputDir + f"sample{s}.wav", sample, sampleRate)
    s += 1

