"""
Description: Train file to train the beat tracking model on ballroom dataset
"""



import glob
import os
import numpy as np
from torch.utils.data import random_split, DataLoader
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam, lr_scheduler
from torch import device, save
from model import Beat_tracking
from get_dataset import ballroom



    


def load_dataset(spectrogram_dir, label_dir, downbeats=False):
    """
     loading the ballroom dataset with spectrogram and labels
    """    
    dataset = ballroom(spectrogram_dir, label_dir, downbeats=downbeats)
    return dataset


def split_dataset(dataset, validation_split, test_split):
    """
    Spliting the dataset into train, test and validation
    """    
    dataset_length = len(dataset)
    test_count = int(dataset_length * test_split)\
        if test_split is not None else 0
    val_count = int(dataset_length * validation_split)
    train_count = dataset_length - (test_count + val_count)
    return random_split(dataset, (train_count, val_count, test_count))

### Train function to train model
def train_model(loader,optimiser,model):
    running_loss = 0
    model.train()
    for i, batch in enumerate(loader):
        optimiser.zero_grad()
        input = batch["spectrogram"]
        label= batch["target"]
        out = model(input)
        loss = criterion(out, label)
        loss.backward()
        optimiser.step()

        batch_loss = loss.item()
        running_loss += batch_loss
    
    return running_loss/len(loader)
#Validation function
def val_model(loader,model):
    running_loss = 0
    model.eval()
    for i, batch in enumerate(loader):
        
        input = batch["spectrogram"]
        label= batch["target"]
        out = model(input)
        loss = criterion(out, label)
    

        batch_loss = loss.item()
        running_loss += batch_loss
    
    return running_loss/len(loader)      






if __name__ == "__main__" :
    
    ##### Data directory path #########
    wav_dir = "./wav_dir/"
    spectrogram_dir = "/Users/harshita/Documents/MODULES/Music_informatics/Assignment/Assignment_1/beat_tracking/spectrogram/"
    label_dir = "/Users/harshita/Documents/MODULES/Music_informatics/Week_4/BallroomAnnotations/"
    label_dir_test = ""
    downbeats = True
    checkpoints_path = "./checkpoints/"

    #### Parameters for training #####
    batch_size = 1
    num_workers = 4
    learning_rate = 0.01
    epochs = 200
    cuda_device = 0
    
    # spectrogram, beat_vector = get_labels_spectrogram(label_dir, spectrogram_dir,wav_dir, downbeats)
    dataset = load_dataset( spectrogram_dir,label_dir, downbeats)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.20, 0)
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader =  DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader =  DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    model = Beat_tracking()
    
    # if cuda_device is not None:
    #     model.cuda(cuda_device)

    criterion = BCELoss()
    optimiser = Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimiser, 'min', factor=0.2)

    min_loss = 100
    for epoch in range(epochs):
        train_loss = train_model(train_loader, optimiser, model)
        val_loss =  val_model(val_loader,model)
        print("Epoch, Train, Val", epoch, train_loss, val_loss)
        if val_loss < min_loss:
            save(model.state_dict(), checkpoints_path + str(epoch) + ".pth")
            min_loss = val_loss
            



