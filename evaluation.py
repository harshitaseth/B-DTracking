"""
File use to evaluate the model on single audio file.
To run this: 
from evaluation import beattracking_output
predicted_beats, predicted_downbeats = beattracking_output(path_of_audio_file)
"""
import pickle
import torch
import librosa
import numpy as np
from model import Beat_tracking
from madmom.features import DBNBeatTrackingProcessor
import mir_eval



model = Beat_tracking()
checkpoint_path = "./Checkpoints/conv_layer_traindataset/"
model.load_state_dict(torch.load(checkpoint_path + "best.pth",map_location=torch.device('cpu')))
model.eval()
sr =22050
hop_length = 220
#Using DBN for post processing
dbn_beat = DBNBeatTrackingProcessor(min_bpm=55,max_bpm=215,transition_lambda=100,fps=(sr/hop_length), online=True)
dbn_downbeat = DBNBeatTrackingProcessor( min_bpm=10, max_bpm=75, transition_lambda=100, fps=(sr/hop_length), online=True)


def create_spectrogram(file_path, n_fft, hop_length_in_seconds, n_mels):
    """
    extract spectrogram and trim it to the model input size
    """
    
    x, sr = librosa.load(file_path)
    hop_length_in_samples = int(np.floor(hop_length_in_seconds * sr))
    spec = librosa.feature.melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length_in_samples, n_mels=n_mels)
    mag_spec = np.abs(spec)[:,:3000] #triming the spectrogram

    return mag_spec

def beattracking_output(file_path):
    """
    take the audio input file and returns  predicted beats and predicted downbeats
    """
    n_fft = 2048
    hop_length_in_seconds = 0.01
    n_mels = 81
    mag_spec = create_spectrogram(file_path,n_fft, hop_length_in_seconds, n_mels)
    input = torch.from_numpy(np.expand_dims(mag_spec.T, axis=0)).unsqueeze(0).float()
    output = model(input)
    beat_activations = output[0][0].detach().numpy()
    downbeat_activations = output[0][1].detach().numpy()

    dbn_beat.reset()
    predicted_beats = dbn_beat.process_offline(beat_activations)
    dbn_downbeat.reset()
    predicted_downbeats = dbn_downbeat.process_offline(downbeat_activations)

    return predicted_beats, predicted_downbeats




if __name__ == "__main__":
    file_path = '/Users/harshita/Documents/MODULES/Music_informatics/Week_4/BallroomData/Jive/Media-106017.wav'
    file_path = "/Users/harshita/Documents/MODULES/Music_informatics/Week_4/BallroomData/Tango/Albums-StrictlyDancing_Tango-03.wav"
    # file_path = "/Users/harshita/Documents/MODULES/Music_informatics/Week_4/BallroomData/Tango/Albums-StrictlyDancing_Tango-08.wav"
    n_fft = 2048
    hop_length_in_seconds = 0.01
    n_mels = 81
    mag_spec = create_spectrogram(file_path,n_fft, hop_length_in_seconds, n_mels)
    mag_spech = torch.from_numpy(np.expand_dims(mag_spec.T, axis=0)).unsqueeze(0)
    beats, downbeats = beattracking_output(file_path)
    results = pickle.load(open("sample_results.pkl","rb"))
    out = results['/Users/harshita/Documents/MODULES/Music_informatics/Assignment/Assignment_1/Full_datasets/test_annotations/'+ file_path.split("/")[-1].split(".")[0]+".beats"]
    beats_gt = out["beats"]["gt"]
    down_gt = out["downbeats"]["gt"]
    import pdb;pdb.set_trace()
    scores = mir_eval.beat.evaluate(beats_gt, beats)
    scores = mir_eval.beat.evaluate(down_gt, downbeats)