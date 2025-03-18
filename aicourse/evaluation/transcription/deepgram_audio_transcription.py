import asyncio
import os
from pathlib import Path
import time as t

from deepgram import DeepgramClient, FileSource, PrerecordedOptions
from dotenv import load_dotenv

PREFIXES = [
    'Chat With Your PDFs： Part 1 - An End to End LangChain Tutorial',
    'Chat With Your PDFs： Part 2 - Frontend - An End to End LangChain Tutorial',
    'Unlock Advanced RAG Secrets： LangChain Tutorial Finale',
    'Unlock the Power of LangChain： Deploying to Production Made Easy'
]
load_dotenv()

async def main(prefixes):
    # Get the current script's directory
    script_dir = Path(__file__).parent
    audio_dir = script_dir.parent / 'audio'
    
    for prefix in prefixes:
        filename = audio_dir / f'{prefix}.mp3'
        deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

        print("Currently transcribing", filename)

        #transcription
        with open(filename, 'rb') as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data
        }

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True
        )

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        with open(f'{prefix}.txt', 'w', encoding='utf8') as file:
            transcript = response.results.channels[0].alternatives[0].transcript
            file.write(transcript)

        print(f"DONE WITH {prefix}.mp3")

asyncio.run(main(PREFIXES))
