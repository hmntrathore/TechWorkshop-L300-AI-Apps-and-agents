import os
import base64
from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np
import time
from urllib.parse import urlparse
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
# Load environment variables (Azure endpoint, deployment, keys, etc.)
load_dotenv(override=True)


def _normalize_azure_endpoint(endpoint: str) -> str:
    """Normalize project-scoped endpoints to the base Azure AI endpoint."""
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("gpt_endpoint is invalid; expected a full https URL")
    return f"{parsed.scheme}://{parsed.netloc}"


def _build_client() -> tuple[AzureOpenAI, str]:
    # Reload env each call to avoid stale process env values.
    load_dotenv(override=True)

    endpoint = os.getenv("gpt_endpoint")
    deployment = os.getenv("gpt_deployment")
    api_version = os.getenv("gpt_api_version", "2024-05-01-preview")

    if not endpoint or not deployment:
        raise ValueError("gpt_endpoint and gpt_deployment must be set")

    normalized_endpoint = _normalize_azure_endpoint(endpoint)
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")

    client = AzureOpenAI(
        azure_endpoint=normalized_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
    )
    return client, deployment


def generate_response(text_input):
    start_time = time.time()
    """
    Input:
        text_input (str): The user's chat input.

    Output:
        response (str): A Markdown-formatted response from the agent.
    """

    # Prepare the full chat prompt with system and user messages
    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """You are a helpful assistant working for Zava, a company that specializes in offering products to assist homeowners with do-it-yourself projects.
                        Respond to customer inquiries with relevant product recommendations and DIY tips. If a customer asks for paint, suggest one of the following three colors: blue, green, and white.
                        If a customer asks for something not related to a DIY project, politely inform them that you can only assist with DIY-related inquiries.
                        Zava has a variety of store locations across the country. If a customer asks about store availability, direct the customer to the Miami store.
                    """
                }
            ]
        },
        {"role": "user", "content": text_input}
    ]

    # Call Azure OpenAI chat API
    client, deployment = _build_client()
    completion = client.chat.completions.create(
        model=deployment,
        messages=chat_prompt,
        max_completion_tokens=10000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )
    end_sum = time.time()
    print(f"generate_response Execution Time: {end_sum - start_time} seconds")
    # Return response content
    return completion.choices[0].message.content
