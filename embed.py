import os
import pandas as pd
from openai import OpenAI

def get_text_embeddings(csv_file, column_name, max_length=8191):
    """
    Read a CSV file with a column of text, split the text into tokens, get text embeddings using OpenAI,
    and write the embeddings to a Pandas DataFrame.

    Args:
        csv_file (str): Path to the CSV file.
        column_name (str): Name of the column containing the text.
        max_length (int, optional): Maximum length of each text. Defaults to 512.

    Returns:
        pandas.DataFrame: DataFrame with the text embeddings.
    """
    client = OpenAI()
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Initialize an empty list to store the embeddings
    embeddings = []

    # Set up OpenAI API credentials
    OpenAI.api_key = os.getenv('OPENAI_API_KEY')

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Get the text from the specified column
        text = row[column_name]

        # Split the text into tokens
        tokens = text.split()

        # Truncate or pad the tokens to the maximum length
        tokens = tokens[:max_length]

        # Join the tokens back into a single string
        truncated_text = ' '.join(tokens)

        # Get the text embeddings using OpenAI
        embedding = client.embeddings.create(input=[truncated_text], model="text-embedding-3-large", dimensions=3072).data[0].embedding

        # Append the embedding to the list
        embeddings.append(embedding)

    # Create a new DataFrame with the embeddings
    embeddings_df = pd.DataFrame(embeddings)

    return embeddings_df


if __name__ == "__main__":
    # Get the text embeddings檔案位置和要的column
    embeddings_df = get_text_embeddings('data/QP.csv', 'Prompt')

    # Write the embeddings to a CSV file
    embeddings_df.to_csv('data/embeddingsV3_3072.csv', index=False)
