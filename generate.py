import argparse

from playgenie.scripts.inference import inference
import config
from playgenie.utils.logging import get_logger
from playgenie.utils.qdrant_search import QdrantSearch

logger = get_logger(__name__)
parser = argparse.ArgumentParser(
                    prog='Play Genie',
                    description='VAE model inference')
search_client = QdrantSearch()

def main():
    args = parser.parse_args()

    n_songs = args.n_songs
    first_song = args.first_song

    logger.info(f"Inferencing..")

    inference(search_client,n_songs, first_song)


if __name__ == "__main__":
    parser.add_argument('--first_song', help='The name of a song',default=None)
    parser.add_argument('--n_songs', help='Number of recommendations to make', default=10)
    try:
        main()
    except Exception as e:
        raise e
