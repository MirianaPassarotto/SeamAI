import os
import torch
import torchaudio
import torchaudio.transforms
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import random
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm 
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

g = torch.Generator()
g.manual_seed(seed)

def seedWorker(workerID): #For determinism
    workerSeed = seed + workerID
    np.random.seed(workerSeed)
    random.seed(workerSeed)
    torch.manual_seed(workerSeed)


#Create a custom dataset
class AudioDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.dataDir = dir
        self.labelMap = {}
        self.labels = []
        self.audioFiles = []

        #Loop through all folders in the dataset
        for idx, label in enumerate(sorted(os.listdir(dir))):
            labelPath = os.path.join(dir, label)
            if(os.path.isdir(labelPath)):
                self.labelMap[label] = idx
                
                #Loop through all files in subfolder (label) and append
                for file in os.listdir(labelPath):
                    if file.endswith(".wav"): 
                        self.audioFiles.append(os.path.join(labelPath, file))
                        self.labels.append(idx)



        self.transform = transform

    #Next two methods are necessary, inheriting from Dataset

    #Return length of dataset
    def __len__(self):
        return len(self.audioFiles)

    #Returns single item/sample and corresponding label of dataset
    def __getitem__(self, index):
        fileName = self.audioFiles[index]
        label = self.labels[index]

        data, sr = torchaudio.load(fileName)

        #Way to handle the last clip from the split being less than 10 seconds
        expectedDur = sr * 5
        if data.shape[1] < expectedDur:
            return None  

        if self.transform: 
            data = self.transform(data)

        return data, label

#Class to transform subsets so that two subsets can have different transforms
class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = [idx for idx in indices if dataset[idx] is not None]
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data, label = self.dataset[self.indices[idx]]
        tdata = self.transform(data)
        return tdata, label

def normalizeSpectrogram(spectrogram):
    mean = spectrogram.mean()
    std = spectrogram.std()
    return (spectrogram - mean) / (std + 1e-6)


#Function to randomly transform the samples, data augmentation
def randomTransform(wave):

    #Applied directly on the audio
    wave = T.Vol(gain=random.uniform(0, 10))(wave) 

    #Convert to spectrogram and apply two more
    spectrogram = MelSpectrogram(n_mels=80)(wave)
    spectrogram = T.TimeMasking(time_mask_param=random.randint(10, 100))(spectrogram) 
    spectrogram = T.FrequencyMasking(freq_mask_param=random.randint(5, 30))(spectrogram) 
    spectrogram = normalizeSpectrogram(spectrogram)

    return spectrogram.squeeze(0) 
        
#Load the full dataset
dataset = AudioDataset("misc/husqvarnaDataSplit5s/")

#Split dataset by index
labels = torch.tensor(dataset.labels)
trainIndices, testIndices = train_test_split(
    list(range(len(dataset))),
    test_size=0.3,
    stratify=labels, 
    random_state=seed
)

#Get the two sets of data, with different transforms
trainDataset = TransformedSubset(dataset, trainIndices, randomTransform)
testDataset = TransformedSubset(dataset, testIndices, lambda data: normalizeSpectrogram(MelSpectrogram(n_mels=80)(data).squeeze(0)))

#Define dataloaders
trainDataloader = DataLoader(trainDataset, batch_size=16, shuffle=True, num_workers=4, worker_init_fn=seedWorker, generator=g)
testDataloader = DataLoader(testDataset, batch_size=16, shuffle=True, num_workers=4, worker_init_fn=seedWorker, generator=g)

#Load model
num_classes = len(set(dataset.labels))  
model = timm.create_model("efficientnet_lite0", pretrained=True, in_chans=1, num_classes=num_classes) #Works, good accuracy
#model = timm.create_model("mobilenetv3_large_100", pretrained=True, in_chans=1, num_classes=num_classes) #Decent accuracy
#model = timm.create_model("mobilenetv3_small_100", pretrained=True, in_chans=1, num_classes=num_classes) #Poor accuracy
#model = timm.create_model("ghostnet_100", pretrained=True, in_chans=1, num_classes=num_classes) #Good accuracy

#Put on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

epochs = 20 

#Train the model
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in trainDataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainDataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")

#Test the model
model.eval()
total_loss, correct, total = 0, 0, 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in testDataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)  
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = total_loss / len(testDataloader)
test_acc = 100 * correct / total
print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

#Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dataset.labelMap.keys(), yticklabels=dataset.labelMap.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(all_labels, all_preds, target_names=dataset.labelMap.keys()))
