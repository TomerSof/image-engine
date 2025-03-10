
import time
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from utility import Policy  # Add this import at the top
key = "ssAjxUjyGCAl6MpIyocz1LeuIfNa7pDg"


def classify_text_objects(textObjs, policy: Policy):
    client = Mistral(api_key=key)
    model = "mistral-small-latest"
    max_retries = 3
    base_delay = 1  # seconds

    for obj in textObjs:
        for attempt in range(max_retries):
            try:
                prompt = f"""Given the text: "{obj.text}"
                First, check if this text matches the pattern: "{policy.pattern}"
                If it matches, respond with: "MATCH:{policy.name}"
                If it doesn't match, respond with "TYPE:" followed by a brief (1-3 words) description of what type of information this is.
                Respond with only one line containing either MATCH: or TYPE:"""

                response = client.chat.complete(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a classification assistant. Provide concise, specific classifications."},
                        {"role": "user", "content": prompt},
                    ]
                )

                if response.choices:
                    result = response.choices[0].message.content.strip()
                    if result.startswith("MATCH:"):
                        obj.objectType = policy.pattern
                        obj.isSensitive = True
                    elif result.startswith("TYPE:"):
                        obj.objectType = result.split(":")[1].strip()
                        obj.isSensitive = False
                break  # Success, exit retry loop

            except SDKError as e:
                if e.status_code == 429:  # Rate limit error
                    if attempt < max_retries - 1:  # Don't sleep on last attempt
                        delay = base_delay * (attempt + 1)  # Exponential backoff
                        print(f"Rate limit hit, waiting {delay} seconds...")
                        time.sleep(delay)
                    continue
                raise  # Re-raise if it's not a rate limit error

    return textObjs
