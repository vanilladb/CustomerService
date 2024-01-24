import os
from openai import OpenAI
client = OpenAI()
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

input='hey flora! Im having some issue with password! can you reset it for me?'

response = client.embeddings.create(
    input=input,
    model="text-embedding-ada-002"
)
question = response.data[0].embedding
print(question)