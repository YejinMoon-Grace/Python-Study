import requests
from bs4 import BeautifulSoup

def get_lyrics(song_id:str) -> dict:
    """
    song_id : 노래 ID
    return : {'title' : '노래제목', 'artist' : '가수', 'lyrics' : '가사'}
    """

    # url 생성
    url = f"https://www.melon.com/song/detail.htm?songId={song_id}"
    head = {"user-agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0"}
    
    # lyrics 생성
    lyrics = requests.get(url, headers=head).text
    bs = BeautifulSoup(lyrics)
    lyrics = BeautifulSoup(str(bs.find("div", id="d_video_summary")).replace("<br/>", "\n")).text.strip()

    # title 생성
    title = bs.find("div", class_="entry").find("div", class_="song_name").text.strip().replace("곡명", "").strip()

    # artist 생성
    artist = bs.find("div", class_="entry").find('a')['title']
    
    return {
        "title" : title,
        "artist" : artist,
        "lyrics" : lyrics
    }
