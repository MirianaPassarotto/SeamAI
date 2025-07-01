import librosa
import numpy as np
import os
import pandas as pd
import pickle


def rawDataToDataframe(audioPath, imuPath, label, clipDuration = 5, seqLength = 5):
    datasetArray = []
    audioClips = []
    imuChunks = []
    
    #Load and split audio
    audio, sr = librosa.load(audioPath, sr=None)
    numClips = len(audio) // (clipDuration * sr)
    audioClips = [audio[i*clipDuration*sr : (i+1)*clipDuration*sr] for i in range(numClips)]
        
    #Process IMU 
    df = pd.read_csv(imuPath)
    t0 = df["timestamp"][0]
    df["timestamp"] = df["timestamp"].apply(lambda x: (x - t0)) #Offset the time so it begins at 0

    #Chunk IMU data to match the audio length
    imuChunks = [df.loc[(df["timestamp"] >= i * clipDuration) & (df["timestamp"] < (i + 1) * clipDuration), ["roll", "pitch", "yaw"]].values for i in range(len(audioClips))]

    #Create dataset
    for i in range(len(audioClips)):
        audio = librosa.feature.melspectrogram(y=audioClips[i], sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        audio = librosa.power_to_db(audio, ref=np.max)
        imu = imuChunks[i]
        
        if len(imu) < clipDuration:
            if clipDuration == 1:  #Special handling when clipDuration is 1
                if i < len(audioClips) - 1 and i > 0:                
                  newData = np.mean([imuChunks[i - 1][0], imuChunks[i + 1][0]], axis=0)
                elif i == 0: #If first value is missing
                    newData = imuChunks[i + 1][0] #Use next point
                else: #If last value is missing
                    newData = imuChunks[i - 1][0]
                imu = np.append(imu, [newData], axis=0)

            #Handle missing IMU data, assume linearity and simply get a new value by averaging
            elif len(imu) == clipDuration - 1:
                if i < len(audioClips) - 1:                
                    newData = np.array([np.mean(j) for j in zip(imu[clipDuration - 2], imuChunks[i + 1][0])])
                    imu = np.append(imu, [newData], axis=0)
                else:
                    newData = np.array([np.mean(j) for j in zip(imu[clipDuration - 2], imuChunks[i - 1][0])])
                    imu = np.append(imu, [newData], axis=0)
            else: #Skip samples where IMU data is too empty
                continue
            
        datasetArray.append([audio, imu, label])

    #Loop through dataset and create longer IMU sequences
    seqLength = 5
    imuArray = [row[1] for row in datasetArray]
    for i in range(len(datasetArray)):
        newEntry = np.empty(seqLength, dtype=object)
        newEntry[0] = datasetArray[i][1]
        for j in range(1, seqLength):
            if i - j < 0: #Zero-pad in the beginning
                newEntry[j] = np.zeros((clipDuration, 3))
            else:
                newEntry[j] = imuArray[i - j]

        newEntry = np.stack(newEntry)
        finalShape = (seqLength * clipDuration, 3)
        datasetArray[i][1] = newEntry[::-1].reshape(finalShape)

    #Return dataframe for current file pair
    dataFrame = pd.DataFrame(datasetArray, columns=["audio", "imu", "label"])
    return dataFrame

if __name__ == "__main__":
    
    dataframes = []
    datasetDir = "misc/mowerDataset"
    labels = os.listdir(datasetDir)

    for label in labels:
        labelPath = os.path.join(datasetDir, label)
        sampleFolders = os.listdir(labelPath)

        for sample in sampleFolders:
            samplePath = os.path.join(labelPath, sample)
            
            files = os.listdir(samplePath)
            audioFile = next((f for f in files if f.endswith('.wav')), None)
            imuFile = next((f for f in files if 'imu' in f.lower()), None)

            if audioFile and imuFile:
                audioPath = os.path.join(samplePath, audioFile)
                imuPath = os.path.join(samplePath, imuFile)
            
                dataframes.append(rawDataToDataframe(audioPath, imuPath, label))
            
            print(samplePath)
            print(label)

    dataset = pd.concat(dataframes, ignore_index=True)
    dataset.to_pickle('misc/mowerModel/mowerDataFin5seconds.pkl')