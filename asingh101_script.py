import os

os.environ["OPENAI_API_KEY"] = "Your_API_Key"
from openai import OpenAI

    # Initialize the OpenAI client
client = OpenAI()

    # Define prompt
prompt_text = "Write a short, creative story about a robot discovering a flower for the first time."

try:
        # Send the prompt to gpt-5-mini
        response = client.chat.completions.create(
            model="gpt-5-mini",  
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=150  
        )

        # Print the response content
        print(response.choices[0].message.content)

except Exception as e:
        print(f"An error occurred: {e}")
