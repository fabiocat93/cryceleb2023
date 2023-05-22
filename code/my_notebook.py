#!/usr/bin/env python
# coding: utf-8

# args
# In[1]:
import argparse
parser = argparse.ArgumentParser(description='Parsing the args of the python script...')

# Add the command-line arguments
parser.add_argument('--device', type=str, help='Device parameter')
parser.add_argument('--encoder', type=str, help='Encoder parameter')
# ecapa-voxceleb-ft-cryceleb (https://huggingface.co/Ubenwa/ecapa-voxceleb-ft-cryceleb, https://huggingface.co/datasets/Ubenwa/CryCeleb2023),
# spkrec-ecapa-voxceleb (https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
# human_cochleagram (https://github.com/mcdermottLab/pycochleagram/tree/master)
# log-mel-spectrogram
# pyannote-embedding
# serab_byols
# apc
# tera
# hubert # 49 layers
# wav2vec2 # 25 layers
# data2vec2 # 25 layers
# bookbot-wav2vec2-adult-child-cls # 13 layers
parser.add_argument('--metric', type=str, help='Metric parameter')
# 'cosine' # 'cosine', 'euclidean', 'manhattan' (https://medium.com/@gshriya195/top-5-distance-similarity-measures-implementation-in-machine-learning-1f68b9ecb0a3, https://www.analyticsvidhya.com/blog/2020/02/4-types-of-distance-metrics-in-machine-learning/#h-minkowski-distance)

# Parse the arguments
args = parser.parse_args()

encoder_name = args.encoder
my_device = args.device
metric = args.metric

# Print the encoder and decoder values
print(f'encoder_name: {encoder_name}')
print(f'my_device: {my_device}')
print(f'metric: {metric}')

# ### Init
sample_rate = 16000
hf_token = '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' #@param {type:"string"}


# ### Imports
import speechbrain as sb
from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.metric_stats import EER
import pandas as pd
import numpy as np
import torch
import pycochleagram.cochleagram as cgram
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from tqdm.notebook import tqdm
import pickle
import os
import pathlib
from torchmetrics.functional import pairwise_euclidean_distance, pairwise_manhattan_distance
from csv import writer
import torchaudio.transforms as T
from pyannote.audio import Model
from pyannote.audio import Inference
import serab_byols
import s3prl.hub as s3hub
#from transformers import Wav2Vec2Model, HubertModel, Data2VecAudioModel
from transformers import AutoProcessor, AutoModelForAudioClassification

# ### Data

# read metadata
metadata = pd.read_csv('../data/metadata/metadata.csv', dtype={'baby_id':str, 'chronological_index':str})
metadata['file_name'] = '../data/' + metadata['file_name']
dev_metadata = metadata.loc[metadata['split']=='dev'].copy()
# read sample submission
sample_submission = pd.read_csv("../data/metadata/sample_submission.csv") # scores are unfiorm random
# read verification pairs
dev_pairs = pd.read_csv("../data/metadata/dev_pairs.csv", dtype={'baby_id_B':str, 'baby_id_D':str})
test_pairs = pd.read_csv("../data/metadata/test_pairs.csv")

#display(metadata.head().style.set_caption("metadata").set_table_styles([{'selector': 'caption','props': [('font-size', '20px')]}]))
#display(dev_pairs.head().style.set_caption("dev_pairs").set_table_styles([{'selector': 'caption','props': [('font-size', '20px')]}]))
#display(test_pairs.head().style.set_caption("test_pairs").set_table_styles([{'selector': 'caption','props': [('font-size', '20px')]}]))
#display(sample_submission.head().style.set_caption("sample_submission").set_table_styles([{'selector': 'caption','props': [('font-size', '20px')]}]))


# ### Verify Pairs

# One way to verify if both pairs come from the same baby is to concatenate all the segments for each pair, compute the embedding of the concatenated cry, and compute the cosine similarity between the embeddings.

if encoder_name == 'ecapa-voxceleb-ft-cryceleb':
    encoder = SpeakerRecognition.from_hparams(
        source="Ubenwa/ecapa-voxceleb-ft-cryceleb",
        savedir=f"../data/models/ecapa-voxceleb-ft-cryceleb",
        run_opts={"device":my_device} #comment out if no GPU available
    ).to(my_device)
    def extract_embeddings(data):
        return encoder.encode_batch(torch.tensor(data).to(my_device), normalize=False)

elif encoder_name == 'spkrec-ecapa-voxceleb':
    encoder = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=f"../data/models/spkrec-ecapa-voxceleb",
        run_opts={"device":my_device} #comment out if no GPU available
    ).to(my_device)
    def extract_embeddings(data):
        return encoder.encode_batch(torch.tensor(data).to(my_device), normalize=False)

elif encoder_name == 'human_cochleagram':
    encoder = cgram.human_cochleagram
    def extract_embeddings(data):
        cochleagram = encoder(data, sample_rate, strict=False, n=40)
        embeddings = (torch.from_numpy(cochleagram) + torch.finfo().eps).log().squeeze(0)
        embeddings = embeddings.mean(1) + embeddings.amax(1)
        embeddings = np.squeeze(embeddings.cpu().detach().numpy())
        return embeddings

elif encoder_name == 'log-mel-spectrogram':
    encoder = T.MelSpectrogram(
                        sample_rate=sample_rate,
                        n_fft=4096,
                        win_length=None,
                        hop_length=512,
                        n_mels=128,
                        f_min=5,
                        f_max=20000,
                        power=2,
                        ).to(my_device)
    def extract_embeddings(data):
        mel_spectrogram = encoder(torch.tensor(data).to(my_device))
        embeddings = (mel_spectrogram + torch.finfo().eps).log().squeeze(0)
        embeddings = embeddings.mean(1) + embeddings.amax(1)
        embeddings = np.squeeze(embeddings.cpu().detach().numpy())
        return embeddings


elif encoder_name == 'pyannote-embedding':
    encoder = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token).to(my_device)
    inference = Inference(encoder, window="whole")
    def extract_embeddings(data):
        return inference({"waveform": torch.unsqueeze(torch.tensor(data), 0).to(my_device), "sample_rate": sample_rate})

elif encoder_name == 'serab_byols':
    encoder_name = 'cvt'
    checkpoint_path = serab_byols.__file__.replace('serab_byols/__init__.py', '') + "checkpoints/cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-osandbyolaloss6373-e100-bs256-lr0003-rs42.pth"
    cfg_path = serab_byols.__file__.replace('serab_byols/__init__.py', 'serab_byols/config.yaml')
    # Load model with weights - located in the root directory of this repo
    encoder = serab_byols.load_model(checkpoint_path, encoder_name, cfg_path).to(my_device)
    def extract_embeddings(data):
        return serab_byols.get_scene_embeddings(torch.unsqueeze(torch.tensor(data), 0).to(my_device), encoder, cfg_path)

elif encoder_name == 'apc' or encoder_name == 'tera':
    encoder = getattr(s3hub, encoder_name)().to(my_device)
    encoder.eval()
    def extract_embeddings(data):
        embeddings = encoder(torch.unsqueeze(torch.tensor(data), 0).to(my_device))["last_hidden_state"]
        embeddings = embeddings.mean(1) + embeddings.amax(1)
        embeddings = np.squeeze(embeddings.cpu().detach().numpy())
        return embeddings

elif encoder_name.startswith('hubert'):
    layer_number = int(encoder_name.split('_')[1])
    weights_file = 'facebook/hubert-xlarge-ll60k'
    pathlib.Path('../data/models/huggingface/').mkdir(parents=True, exist_ok=True)
    encoder = AutoModelForAudioClassification.from_pretrained(weights_file, cache_dir='../data/models/huggingface/').to(my_device)
    def extract_embeddings(data):
        output = encoder(torch.unsqueeze(torch.tensor(data), 0).to(my_device), output_hidden_states=True)
        embeddings = output.hidden_states[layer_number]
        embeddings = embeddings.mean(1) + embeddings.amax(1)
        embeddings = np.squeeze(embeddings.cpu().detach().numpy())
        return embeddings

elif encoder_name.startswith('wav2vec2'):
    layer_number = int(encoder_name.split('_')[1])
    weights_file = 'facebook/wav2vec2-large-960h-lv60-self'
    pathlib.Path('../data/models/huggingface/').mkdir(parents=True, exist_ok=True)
    encoder = AutoModelForAudioClassification.from_pretrained(weights_file, cache_dir='../data/models/huggingface/').to(my_device)
    def extract_embeddings(data):
        output = encoder(torch.unsqueeze(torch.tensor(data), 0).to(my_device), output_hidden_states=True)
        embeddings = output.hidden_states[layer_number]
        embeddings = embeddings.mean(1) + embeddings.amax(1)
        embeddings = np.squeeze(embeddings.cpu().detach().numpy())
        return embeddings

elif encoder_name.startswith('data2vec2'):
    layer_number = int(encoder_name.split('_')[1])
    weights_file = 'facebook/data2vec-audio-large-960h'
    pathlib.Path('../data/models/huggingface/').mkdir(parents=True, exist_ok=True)
    encoder = AutoModelForAudioClassification.from_pretrained(weights_file, cache_dir='../data/models/huggingface/').to(my_device)
    def extract_embeddings(data):
        output = encoder(torch.unsqueeze(torch.tensor(data), 0).to(my_device), output_hidden_states=True)
        embeddings = output.hidden_states[layer_number]
        embeddings = embeddings.mean(1) + embeddings.amax(1)
        embeddings = np.squeeze(embeddings.cpu().detach().numpy())
        return embeddings

elif encoder_name.startswith('bookbot-wav2vec2-adult-child-cls'):
    layer_number = int(encoder_name.split('_')[1])
    weights_file = 'bookbot/wav2vec2-adult-child-cls'
    pathlib.Path('../data/models/huggingface/').mkdir(parents=True, exist_ok=True)
    encoder = AutoModelForAudioClassification.from_pretrained(weights_file, cache_dir='../data/models/huggingface/').to(my_device)
    def extract_embeddings(data):
        output = encoder(torch.unsqueeze(torch.tensor(data), 0).to(my_device), output_hidden_states=True)
        embeddings = output.hidden_states[layer_number]
        embeddings = embeddings.mean(1) + embeddings.amax(1)
        embeddings = np.squeeze(embeddings.cpu().detach().numpy())
        return embeddings




# #### Compute Encodings
embeddings_file = "../data/embeddings/" + encoder_name + ".pkl"
if not os.path.exists(embeddings_file):
    # read the segments
    dev_metadata['cry'] = dev_metadata.apply(lambda row: read_audio(row['file_name']).numpy(), axis=1)
    # concatenate all segments for each (baby_id, period) group
    cry_dict = pd.DataFrame(dev_metadata.groupby(['baby_id', 'period'])['cry'].agg(lambda x: np.concatenate(x.values)),
                            columns=['cry']).to_dict(orient='index')
    # encode the concatenated cries
    for (baby_id, period), d in tqdm(cry_dict.items()):
        d['cry_encoded'] = extract_embeddings(d['cry'])

    pathlib.Path(os.path.dirname(embeddings_file)).mkdir(parents=True, exist_ok=True)

    # Iterate through the dictionary and move tensors to CPU
    for key in cry_dict:
        for sub_key in cry_dict[key]:
            if isinstance(cry_dict[key][sub_key], torch.Tensor):
                cry_dict[key][sub_key] = cry_dict[key][sub_key].to('cpu')

    with open(embeddings_file, 'wb') as fp:
        pickle.dump(cry_dict, fp)
else:
    with open(embeddings_file, 'rb') as fp:
        cry_dict = pickle.load(fp)


# #### Compute Similarity Between Encodings
if metric == 'cosine':
    def compute_similarity_score(row, cry_dict):
        cos = torch.nn.CosineSimilarity(dim=-1)
        similarity_score = cos(
          torch.tensor(cry_dict[(row['baby_id_B'], 'B')]['cry_encoded']),
          torch.tensor(cry_dict[(row['baby_id_D'], 'D')]['cry_encoded'])
        )
        return similarity_score.item()
elif metric == 'euclidean':
    def compute_similarity_score(row, cry_dict):
        return torch.cdist(
            torch.unsqueeze(torch.tensor(cry_dict[(row['baby_id_B'], 'B')]['cry_encoded']), 0),
            torch.unsqueeze(torch.tensor(cry_dict[(row['baby_id_D'], 'D')]['cry_encoded']), 0),
            p=2
        ).squeeze().item()
elif metric == 'manhattan':
    def compute_similarity_score(row, cry_dict):
        return torch.cdist(
            torch.unsqueeze(torch.tensor(cry_dict[(row['baby_id_B'], 'B')]['cry_encoded']), 0),
            torch.unsqueeze(torch.tensor(cry_dict[(row['baby_id_D'], 'D')]['cry_encoded']), 0),
            p=1
        ).squeeze().item()

dev_pairs['score'] = dev_pairs.apply(lambda row: compute_similarity_score(row=row, cry_dict=cry_dict), axis=1)
#display(dev_pairs.head())

def compute_eer_and_plot_verification_scores(pairs_df):
    ''' pairs_df must have 'score' and 'label' columns'''
    positive_scores = pairs_df.loc[pairs_df['label']==1]['score'].values
    negative_scores = pairs_df.loc[pairs_df['label']==0]['score'].values
    eer, threshold = EER(torch.tensor(positive_scores), torch.tensor(negative_scores))
    ax = sns.histplot(pairs_df, x='score', hue='label', stat='percent', common_norm=False)
    ax.set_title(f'EER={round(eer, 4)} - Thresh={round(threshold, 4)}')
    plt.axvline(x=[threshold], color='red', ls='--');

    # Save figure as SVG
    pathlib.Path('../data/figures/').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'../data/figures/{metric}_{encoder_name}.svg', format='svg')
    # Save figure as PNG
    plt.savefig(f'../data/figures/{metric}_{encoder_name}.png', format='png')
    # Show the figure
    # plt.show()
    return eer, threshold

eer, threshold = compute_eer_and_plot_verification_scores(pairs_df=dev_pairs)

# List that we want to add as a new row
my_list = [encoder_name, metric, eer, threshold]

# Open our existing CSV file in append mode
# Create a file object for this file
pathlib.Path('../data/overall/').mkdir(parents=True, exist_ok=True)
with open('../data/overall/overall.csv', 'a') as f_object:
    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(my_list)

    # Close the file object
    f_object.close()

'''
test_metadata = metadata.loc[metadata['split']=='test'].copy()
# read the segments
test_metadata['cry'] = test_metadata.apply(lambda row: read_audio(row['file_name']).numpy(), axis=1)
# concatenate all segments for each (baby_id, period) group
cry_dict_test = pd.DataFrame(test_metadata.groupby(['baby_id', 'period'])['cry'].agg(lambda x: np.concatenate(x.values)), columns=['cry']).to_dict(orient='index')
# encode the concatenated cries
for (baby_id, period), d in tqdm(cry_dict_test.items()):
  d['cry_encoded'] = extract_embeddings(d['cry'])

# compute cosine similarity between all pairs
test_pairs['score'] = test_pairs.apply(lambda row: compute_similarity_score(row=row, cry_dict=cry_dict_test), axis=1)
#display(test_pairs.head())

#submission must match the 'sample_submission.csv' format exactly
my_submission= test_pairs[['id', 'score']]
my_submission_file = f'../data/results/{metric}_{encoder_name}.csv'
pathlib.Path(os.path.dirname(my_submission_file)).mkdir(parents=True, exist_ok=True)
my_submission.to_csv(my_submission_file, index=False)
#display(my_submission.head())
'''

# You can now download `my_submission.csv` and submit it to the challenge!