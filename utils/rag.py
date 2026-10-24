import os

from groq import Groq


class RAGAssistant:

    def __init__(self):

        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

    def generate_answer(
        self,
        question,
        context,
        language="English"
    ):

        prompt = f"""
You are an AI document assistant.

Answer the user's question using ONLY the
information provided in the context.

If the answer cannot be found in the context,
say:

"I could not find this information in the uploaded documents."

Do not invent facts.

Answer in {language}.

CONTEXT:
{context}

QUESTION:
{question}
"""

        response = self.client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": "You are a precise document question-answering assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=1200
        )

        return response.choices[0].message.content