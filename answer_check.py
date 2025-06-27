import json
from openai import AzureOpenAI
import os
import time


def build_check_prompt(query, answer, context_full):
    return f"""
        Given the following context, question, and answer:
        Question: {query}\n
        User's answer: {answer}\n
        Context:\n{context_full}\n
        Does the answer accurately answer the question based on the context, and does it contain no information that is incorrect or not supported by the context?
        Answer only "yes" or "no".
        """

def gpt_check_answer(query, answer, context, max_retry=5, sleep_sec=2):
    client = AzureOpenAI(
        api_version="YOUR_AZURE_OPENAI_API_VERSION",  # Replace with your actual API version
        azure_endpoint= "YOUR_AZURE_OPENAI_ENDPOINT",  # Replace with your actual endpoint
        api_key="YOUR_AZURE_OPENAI_API_KEY",  # Replace with your actual API key
    )
    prompt = build_check_prompt(query, answer, context)
    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt}
                ],
                model="o4-mini",
                max_completion_tokens=4096,
            )
            content = response.choices[0].message.content.strip()
            if content:  
                return content
        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retry}] Error querying GPT: {e}")
            if attempt < max_retry - 1:
                time.sleep(sleep_sec)
    print(f"All retries failed. Returning 'fail'.")
    return "fail"