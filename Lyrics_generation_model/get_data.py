import requests
from bs4 import BeautifulSoup
import json
import time

class SongScraper:
    def __init__(self, token):
        """
        Initializes the SongScraper class.

        Args:
            token (str): The Genius API token.
        """
        self.GENIUS_API_KEY = token

    def get_lyrics(self, url):
        """
        Retrieves the lyrics of a song from the given URL.

        Args:
            url (str): The URL of the song.

        Returns:
            str: The lyrics of the song.
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        lyrics_div = soup.find('div', class_='Lyrics__Container-sc-1ynbvzw-1 kUgSbL')
        lyrics = lyrics_div.get_text(separator='\n') if lyrics_div else None
        return lyrics

    def get_tags(self, url):
        """
        Retrieves the tags of a song from the given URL.

        Args:
            url (str): The URL of the song.

        Returns:
            str: The tags of the song.
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        tags_div = soup.find('div', class_='SongTags__Container-xixwg3-1 bZsZHM')
        tags = tags_div.get_text(separator=',') if tags_div else None
        return tags

    def get_name(self, url):
        """
        Retrieves the name of the artist from the given URL.

        Args:
            url (str): The URL of the song.

        Returns:
            str: The name of the artist.
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        name_a = soup.find('a', class_="StyledLink-sc-3ea0mt-0 fcVxWP HeaderArtistAndTracklistdesktop__Artist-sc-4vdeb8-1 jhWHLb")
        name = name_a.get_text() if name_a else None
        return name

    def get_songs(self):
        """
        Retrieves songs from a list of artist search URLs.

        Returns:
            list: A list of dictionaries containing song information.
        """
        artist_search_urls = [
            # List of artist search URLs
        ]

        songs = []

        for search_url in artist_search_urls:
            headers = {'Authorization': 'Bearer ' + self.GENIUS_API_KEY}
            response = requests.get(search_url, headers=headers)
            data = response.json()

            for hit in data['response']['hits']:
                song = {}
                song['title'] = hit['result']['title']
                song['url'] = hit['result']['url']

                # Retrieve the lyrics of the song
                lyrics = self.get_lyrics(song['url'])
                song['lyrics'] = lyrics

                # Retrieve the tags of the song
                tags = self.get_tags(song['url'])
                song['category'] = tags

                name = self.get_name(song['url'])
                song['artist'] = name
                songs.append(song)

            time.sleep(1)  # Pause of 1 second between each URL request

        return songs

    def save_to_json(self, songs):
        """
        Saves the song data to a JSON file.

        Args:
            songs (list): A list of dictionaries containing song information.
        """
        with open('songs_data.json', 'w') as f:
            json.dump(songs, f, indent=4)

        print("Data written to songs_data.json")

# Usage
scraper = SongScraper('your_token_here')
songs = scraper.get_songs()
scraper.save_to_json(songs)
