"""
Description: Test file to evaluate the performance of the model on test dataset.
"""





import glob
import os
from unittest import result
import librosa
import numpy as np
from torch.utils.data import random_split, DataLoader
from torch.nn import CrossEntropyLoss
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
    train_dataset, val_dataset, test_dataset = random_split(dataset, (train_count, val_count, test_count))
    out = []
    for i , data in enumerate(test_dataset):
        file_path = data["path"]
        out.append(file_path)

    return random_split(dataset, (train_count, val_count, test_count))


def test(loader,model):
   import torch
   import mir_eval
   from madmom.features import DBNBeatTrackingProcessor
   model.load_state_dict(torch.load(checkpoints_path + "best.pth",map_location=torch.device('cpu')))
   model.eval()
   HOP_LENGTH_IN_SAMPLES = 220
   SR = 22050
   dbn = DBNBeatTrackingProcessor(min_bpm=55,max_bpm=215,transition_lambda=100,fps=(SR / HOP_LENGTH_IN_SAMPLES), online=True)
   downbeat_dbn = DBNBeatTrackingProcessor( min_bpm=10, max_bpm=75, transition_lambda=100, fps=(SR / HOP_LENGTH_IN_SAMPLES), online=True)
   scores_beats = 0
   scores_downbeats = 0
#    thresholds  = [0.005,0.01,0.05,0.1,0.2,0.25,0.3,0.35]
   thresholds = [1]
   beats_dict = {}
   downbeats_dict = {}
   results = {}
#    import pdb;pdb.set_trace()
   for thresh in thresholds:
        print("thresh:", thresh)
        scores_beats = 0
        scores_downbeats = 0  
        max_score = 0
        file_path = '/Users/harshita/Documents/MODULES/Music_informatics/Assignment/Assignment_1/Full_datasets/test_annotations/Media-106017.beats'
        f_score_beatout = []
        f_score_downbeatout = []
        for i, batch in enumerate(loader):
                results[batch['path'][0]] = {}
                input = batch["spectrogram"]
                label= batch["target"]
                beats_gt = (batch['beats'].detach().numpy())/SR
                downbeats_gt = (batch['downbeats'].detach().numpy())/SR
                out = model(input)
                beat_activations = out[0][0].detach().numpy()#[5:]
                downbeat_activations = out[0][1].detach().numpy()#[5:]
                # beat_activations[beat_activations >= thresh] = 1
                # beats_indices = np.argwhere(beat_activations ==1).reshape(-1)
                # predicted_beats = librosa.frames_to_time(beats_indices)

                # downbeat_activations[downbeat_activations >= thresh] = 1
                # downbeats_indices = np.argwhere(downbeat_activations ==1).reshape(-1)
                # predicted_downbeats = librosa.frames_to_time(downbeats_indices)
                dbn.reset()
                predicted_beats = dbn.process_offline(beat_activations)
                downbeat_dbn.reset()
                predicted_downbeats = downbeat_dbn.process_offline(downbeat_activations)
                scores = mir_eval.beat.evaluate(beats_gt, predicted_beats)
                f_score_beatout.append(scores["F-measure"])
                scores_beats += scores["F-measure"]
                results[batch['path'][0]]["beats"] = {}
                results[batch['path'][0]]["beats"]["gt"] = beats_gt
                results[batch['path'][0]]["beats"]["pred"] = predicted_beats


                if scores["F-measure"] > max_score:
                    max_score =scores["F-measure"]
                    file_path = batch["path"]
                scores = mir_eval.beat.evaluate(downbeats_gt, predicted_downbeats)
                scores_downbeats += scores["F-measure"]
                f_score_downbeatout.append(scores["F-measure"])
                results[batch['path'][0]]["downbeats"] = {}
                results[batch['path'][0]]["downbeats"]["gt"] = downbeats_gt
                results[batch['path'][0]]["downbeats"]["pred"] = predicted_downbeats
        beats_dict[thresh] = scores_beats/len(loader)
        downbeats_dict[thresh] = scores_downbeats/len(loader)

   return scores_beats/len(loader), scores_downbeats/len(loader)

   




if __name__ == "__main__" :
    
    ##### Data directory path #########
    wav_dir = "./wav_dir/"
    spectrogram_dir = "/Users/harshita/Documents/MODULES/Music_informatics/Assignment/Assignment_1/beat_tracking/spectrogram/"
    label_dir_test = "/Users/harshita/Documents/MODULES/Music_informatics/Assignment/Assignment_1/Full_datasets/test_annotations/"
    downbeats = True
    checkpoints_path = "./Checkpoints/conv_layer_traindataset/"

    #### Parameters for training #####
    batch_size = 1
    num_workers = 4
    learning_rate = 0.01
    epochs = 20
    cuda_device = 0
    
    dataset = load_dataset( spectrogram_dir,label_dir_test, downbeats)

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0, 1)
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader =  DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader =  DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    model = Beat_tracking()
    test(test_loader,model)
    # if cuda_device is not None:
    #     model.cuda(cuda_device)




