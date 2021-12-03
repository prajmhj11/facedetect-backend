import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

auth_manager = SpotifyClientCredentials(client_id='810f0d189436477b9fc9e7c056aa6d33',
                                        client_secret='a6cb9f1608904784acaebcdadcee6dbb')
sp = spotipy.Spotify(auth_manager=auth_manager)


def getTrackIDs(user, playlist_id):
    track_ids = []
    playlist = sp.user_playlist(user, playlist_id)
    for item in playlist['tracks']['items']:
        track = item['track']
        track_ids.append(track['id'])
    return track_ids


def getTrackFeatures(id):
    track_info = sp.track(id)
    # audio_info = sp.audio_features(id)
    artists = []
    id = track_info['id']
    name = track_info['name']
    album = track_info['album']['name']
    for artist in track_info['album']['artists']:
        artists.append(artist['name'])
    release_date = track_info['album']['release_date']
    # length = track_info['duration_ms']
    # explicit = track_info['explicit']
    year = release_date
    # popularity = track_info['popularity']

    track_data = [id, name, album, artists, year]  # release_date, length, popularity
    return track_data


# Code for creating dataframe of feteched playlist

emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}
music_dist = {0: "0l9dAmBrUJLylii66JOsHB?si=e1d97b8404e34343", 1: "1n6cpWo9ant4WguEo91KZh?si=617ea1c66ab6446b",
              2: "4cllEPvFdoX6NIVWPKai9I?si=dfa422af2e8448ef", 3: "0deORnapZgrxFY4nsKr9JA?si=7a5aba992ea14c93",
              4: "4kvSlabrnfRCQWfN0MgtgA?si=b36add73b4a74b3a", 5: "1n6cpWo9ant4WguEo91KZh?si=617ea1c66ab6446b",
              6: "37i9dQZEVXbMDoHDwVN2tF?si=c09391805b6c4651"}

emotion_music_dict = {"angry": "0l9dAmBrUJLylii66JOsHB?si=e1d97b8404e34343", "disgusted": "1n6cpWo9ant4WguEo91KZh?si=617ea1c66ab6446b",
                      "fearful": "4cllEPvFdoX6NIVWPKai9I?si=dfa422af2e8448ef", "happy": "51WkmNDvNKlgDUHM9bSjgO?si=-urqIJyzTz2xPcwG9usTEQ",
                      "neutral": "4kvSlabrnfRCQWfN0MgtgA?si=b36add73b4a74b3a", "sad": "1n6cpWo9ant4WguEo91KZh?si=617ea1c66ab6446b",
                      "surprised": "37i9dQZEVXbMDoHDwVN2tF?si=c09391805b6c4651"}


def refreshPlaylist(emotion):
    if emotion in emotion_music_dict:
        track_ids = getTrackIDs('spotify', emotion_music_dict[emotion])
        print(sp.playlist(emotion_music_dict[emotion]))
        track_list = []
        for i in range(len(track_ids)):
            time.sleep(.3)
            # print(getTrackFeatures(track_ids[1]))
            track_data = getTrackFeatures(track_ids[i])
            track_list.append(track_data)
            df = pd.DataFrame(track_list,
                              columns=['id', 'name', 'album', 'artists', 'year'])  # ,'Release_date','Length','Popularity'
            df.to_csv('songs/' + emotion + '.csv')
        print(emotion + ".csv Generated")
        return True
    else:
        return False


'''
for index in music_dist:
    track_ids = getTrackIDs('spotify', music_dist[index])
    track_list = []
    for i in range(len(track_ids)):
        time.sleep(.3)
        # print(getTrackFeatures(track_ids[1]))
        track_data = getTrackFeatures(track_ids[i])
        track_list.append(track_data)
        df = pd.DataFrame(track_list,
                          columns=['id', 'name', 'album', 'artists', 'year'])  # ,'Release_date','Length','Popularity'
        df.to_csv('songs/' + emotion_dict[index] + '.csv')
    print(emotion_dict[index] + ".csv Generated")

'''
