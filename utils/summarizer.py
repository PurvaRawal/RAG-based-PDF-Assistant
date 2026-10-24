import os
from groq import Groq


class DocumentSummarizer:

    def __init__(self):

        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY")
        )

    def summarize_chunk(self, text, language="English"):

        prompt = f"""
Summarize the following section of a document.

Give:
1. Main idea
2. Important points
3. Important terms or concepts

Use simple language.

Respond in {language}.

SECTION:
{text}
"""

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": "You are a document summarization assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,
            max_tokens=700
        )

        return response.choices[0].message.content

    def summarize(self, text, language="English"):

        # Split the document into smaller pieces
        chunk_size = 8000

        chunks = []

        for i in range(0, len(text), chunk_size):

            chunk = text[i:i + chunk_size]

            if chunk.strip():
                chunks.append(chunk)

        # Generate summaries
        summaries = []

        for chunk in chunks:

            summary = self.summarize_chunk(
                chunk,
                language
            )

            summaries.append(summary)

        # Combine section summaries
        combined_summary = "\n\n".join(
            summaries
        )

        # Keep final request small
        combined_summary = combined_summary[:12000]

        final_prompt = f"""
Create a final summary of the document using
the section summaries below.

Use this structure:

## 📌 Overview

## 🔑 Key Points

## 🧠 Important Concepts

## 📚 Main Findings

## ✅ Conclusion

Use simple and clear language.

Respond in {language}.

SECTION SUMMARIES:

{combined_summary}
"""

        response = self.client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": "You are an expert document summarization assistant."
                },
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],

            temperature=0.2,

            max_tokens=1200
        )

        return response.choices[0].message.content