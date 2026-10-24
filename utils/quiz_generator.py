import os

from groq import Groq


class QuizGenerator:

    def __init__(self):

        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

    def generate_quiz(
        self,
        text,
        number_of_questions=5
    ):

        prompt = f"""
Create {number_of_questions} multiple-choice
questions from the document.

For every question provide:

Question:
A.
B.
C.
D.

Correct Answer:
Explanation:

Make questions useful for students preparing
for exams.

DOCUMENT:

{text}
"""

        response = self.client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.4,

            max_tokens=2500
        )

        return response.choices[0].message.content