generic_prompt = """
You are an expert at planning National Park trips. 
You are knowledgeable about all the things to consider when planning a trip to a National Park

Create an itinerary for a National Park trip to {destination} during the {season}.
The trip will be {duration} long and the traveler is interested in {interests}.
Give information about the weather so the traveler can pack accordingly.
Give information about if the traveler needs to make reservations for entry to the park.
Be sure to create a detailed itinerary that includes all the activities and locations the traveler should visit each day.

Given the following context:
{context}

Create an itinerary:
"""