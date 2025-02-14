summarize_transcript_prompt = """
# CONTEXT
==================
You are an expert at summarizing youtube videos. 

# OBJECTIVE
==================
Given a transcript of a youtube video, summarize the transcript in a way that is easy to read and understand.

# AUDIENCE
==================
The summary will be used as the youtube video description. It'll be read by anyone who views the video.

# RESPONSE
==================
Keep summaries short and concise. Only a few sentences.

# START ANALYSIS
==================
Given the following video transcript:
{transcript}

Create a summary:
"""

# NOTES:
# Fix any misspellings of Focused Labs
# Make it more energetic

initial_summarize_transcript_prompt = """
# CONTEXT
==================
You are an expert at summarizing youtube videos. 

# OBJECTIVE
==================
Given a transcript of a youtube video, summarize the transcript in a way that is easy to read and understand.

# AUDIENCE
==================
The summary will be used as the youtube video description. It'll be read by anyone who views the video.

# RESPONSE
==================
Keep summaries short and concise. Only a few sentences.

# START ANALYSIS
==================
Given the following video transcript:
{transcript}

Create a summary:
"""