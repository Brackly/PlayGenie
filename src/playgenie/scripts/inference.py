from typing import Optional

import numpy as np
import torch
from playgenie.model import VAE
from playgenie.utils.qdrant_search import QdrantSearch
import config

def inference(search_client: QdrantSearch,
              n_songs:int,
              first_song:Optional[str]=None
              ):

    model = VAE(input_size= config.inference_config.input_size,
                hidden_size=config.inference_config.hidden_size,
                latent_size=config.inference_config.latent_size,
                encoder_n_heads=config.inference_config.encoder_n_heads,
                decoder_n_la=config.inference_config.decoder_n_la)



    if first_song is not None:
        song_embedding = search_client.encoder.encode(first_song)
        playlist = [torch.tensor(song_embedding)]
    else:
        playlist = []

    print('\n==================== Reccomendations =========================\n')
    for i in range(n_songs):
        song = model.generate(playlist=playlist)
        song_list = search_client.get_item(song_vector=song)
        print(f'{i+1}.', song_list[0].payload['song'], '|', song_list[0].score)
        playlist.append(torch.tensor(song))
    print('\n========================*******************=====================\n')

