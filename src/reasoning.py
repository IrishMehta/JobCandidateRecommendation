import os
import time
from typing import Optional
from groq import Groq, APIError
from src.utils import get_env_var

try:
    if Groq:
        client = Groq(api_key=get_env_var("GroqAPI"))
    else:
        client = None
except Exception as e:
    client = None


def generate_comprehensive_fit_reasoning(
    job_description: str,
    resume_text: str,
    candidate_name: str,
    max_retries: int = 2,
    retry_delay: int = 5,
) -> str:
    """
    Generates a candidate fit explanation using the Groq API with a retry mechanism.

    Note: The 3-5 second pause between processing different resumes should be
    handled by the calling script that iterates through the list of candidates.

    Args:
        job_description: The full text of the job description.
        resume_text: The full text of the candidate's resume.
        candidate_name: The name of the candidate.
        max_retries: The maximum number of times to retry the API call upon failure.
        retry_delay: The number of seconds to wait between retries.

    Returns:
        A string containing the reasoning, or an error message if all retries fail.
    """
    if not client:
        return "Error: Groq client is not initialized. Please check your API key."

    prompt = f"""
    Based on the following resume and job description, explain in one or two concise sentences 
    why the candidate '{candidate_name}' is a good fit for this role. Synthesize the key 
    qualifications from the resume and connect them directly to the requirements in the job description.

    ## Job Description:
    {job_description}

    ## Resume:
    {resume_text}
    """

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.3,
                max_tokens=500, 
                top_p=1,
                reasoning_effort="low",
                stream=False,
                stop=None
            )
            
            response = completion.choices[0].message.content.strip()
            return response

        except APIError as e:
            pass
        except Exception as e:
            pass

        if attempt < max_retries:
            time.sleep(retry_delay)

    return "Failed to generate reasoning after multiple retries."
