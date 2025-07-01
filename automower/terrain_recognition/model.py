import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm 
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns

seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#Load data from file
with open("misc/mowerModel/mowerDataFin2seconds.pkl", "rb") as file:
    data = pickle.load(file)

#Custom dataset for multimodal data
class MultimodalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.labels, self.uniqueLabels = pd.factorize(self.data['label'])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        labelIdx = self.labels[idx]

        return torch.tensor(sample["audio"], dtype=torch.float32).unsqueeze(0), torch.tensor(sample["imu"], dtype=torch.float32), torch.tensor(labelIdx, dtype=torch.long)

#Create dataset, use chunks so that the model doesnt just remember
chunkSize = 5
data["chunkId"] = data.index // chunkSize

uniqueChunks = data["chunkId"].unique()

#Split the chunk IDs into train/test sets
trainChunks, testChunks = train_test_split(uniqueChunks, test_size=0.2, random_state=seed)

# Create the actual train/test DataFrames by filtering on chunkId
trainData = data[data["chunkId"].isin(trainChunks)].reset_index(drop=True)
testData = data[data["chunkId"].isin(testChunks)].reset_index(drop=True)

trainDataset = MultimodalDataset(trainData)
testDataset = MultimodalDataset(testData)

#Create DataLoader
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, num_workers=0)
testLoader = DataLoader(testDataset, batch_size=32, num_workers=0)

#Class for multimodal model
class MultimodalModel(nn.Module):
    def __init__(self, numClasses):
        super(MultimodalModel, self).__init__()
        
        #Define cnn, remove classification layer
        self.cnn = timm.create_model("efficientnet_lite0", pretrained=True, in_chans=1) 
        self.cnn.fc = nn.Identity() 
        #self.cnnDropout = nn.Dropout(p=0.7)

        #Define lstm
        self.lstm = nn.LSTM(input_size=3, hidden_size=512, num_layers=2, batch_first=True)#, dropout = 0.7)

        #Fully connected layer
        self.fc = nn.Linear(1512, numClasses)
    
    def forward(self, audio, imu):
        cnnOut = self.cnn(audio)
        #cnnOut = self.cnnDropout(cnnOut)

        imuOut, (hn, cn) = self.lstm(imu)
        imuOut = hn[-1]

        #Put the features into a fully connected layer
        combined = torch.cat((cnnOut, imuOut), dim=1)
        return self.fc(combined)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Model
numClasses = len(trainDataset.uniqueLabels)
model = MultimodalModel(numClasses).to(device)

#Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

print("Label to index mapping:")
for idx, label in enumerate(trainDataset.uniqueLabels):
    print(f"{idx}: {label}")

#Training
def train(model, trainLoader, optimizer, criterion, device):
    model.train() 
    runningLoss = 0.0
    correct = 0
    total = 0
    
    for audio, imu, labels in trainLoader:
        audio, imu, labels = audio.to(device), imu.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(audio, imu)
        
        loss = criterion(outputs, labels)
        runningLoss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epochLoss = runningLoss / len(trainLoader)
    epochAcc = 100 * correct / total
    return epochLoss, epochAcc

#Testing
def test(model, testLoader, criterion, device):
    model.eval() 
    runningLoss = 0.0
    correct = 0
    total = 0
    allPreds = []
    allLabels = []
    
    with torch.no_grad():
        for audio, imu, labels in testLoader:

            audio, imu, labels = audio.to(device), imu.to(device), labels.to(device)
            
            outputs = model(audio, imu)

            loss = criterion(outputs, labels)
            runningLoss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            allPreds.extend(predicted.cpu().numpy())
            allLabels.extend(labels.cpu().numpy())
    
    epochLoss = runningLoss / len(testLoader)
    epochAcc = 100 * correct / total
    report = classification_report(allLabels, allPreds, target_names=testDataset.uniqueLabels, output_dict=True)
    return epochLoss, epochAcc, report

#Train and test
numEpochs = 10

for epoch in range(numEpochs):
    trainLoss, trainAcc = train(model, trainLoader, optimizer, criterion, device)

    testLoss, testAcc, testReport = test(model, testLoader, criterion, device)
    
    print(f"Epoch {epoch+1}/{numEpochs}")
    print(f"Train Loss: {trainLoss:.4f}, Train Accuracy: {trainAcc:.2f}%")
    print(f"Test Loss: {testLoss:.4f}, Test Accuracy: {testAcc:.2f}%")
    print(f"Classification Report: \n{testReport}")


torch.save({
    'model_state_dict': model.state_dict(),
    'label_mapping': trainDataset.uniqueLabels.tolist()
}, 'misc/models/model_with_labels.pth')

'''
modelScript = torch.jit.script(model)
modelScript.save('misc/models/multimodalNet.pt')
'''
