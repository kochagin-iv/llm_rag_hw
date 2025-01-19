import os

from constants import MODEL_NAME


def run_mistral(client, user_message, model=MODEL_NAME):
    messages = [{"role": "user", "content": user_message}]
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content
