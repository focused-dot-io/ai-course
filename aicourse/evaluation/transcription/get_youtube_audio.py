import yt_dlp

videos = [
    "https://www.youtube.com/watch?v=UwgZmrRAgQ4",
    "https://www.youtube.com/watch?v=xFWllDS6ZRw",
    "https://www.youtube.com/watch?v=CbBIwVxjdP8",
    "https://www.youtube.com/watch?v=pOuQp-vfuns",
]


def download_audio(link):
    with yt_dlp.YoutubeDL(
        {"extract_audio": True, "format": "bestaudio", "outtmpl": "%(title)s.mp3"}
    ) as video:
        info_dict = video.extract_info(link, download=True)
        video_title = info_dict["title"]
        print(video_title)
        video.download(link)
        print("Successfully Downloaded - see local folder on Google Colab")


def main():
    for video in videos:
        download_audio(video)


if __name__ == "__main__":
    main()
