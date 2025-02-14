import asyncio
import json
import os
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
    for prefix in prefixes:
        filename = "../audio/" + prefix + ".mp3"
        deepgram = DeepgramClient(os.getenv("DEEPGRAM_API_KEY"))

        print("Currently transcribing ", filename)

        # start transcribing
        with open(filename, 'rb') as file:
            buffer_data = file.read()

        # write results
        payload: FileSource = {
            "buffer": buffer_data,
        }
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True
        )

        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

        with open(f'{prefix}.txt', 'w', encoding='utf8') as file:
            transcript = response.results.channels[0].alternatives[0].transcript
            file.write(transcript)

        print(f'DONE WITH {prefix}.mp3')
asyncio.run(main(PREFIXES))
