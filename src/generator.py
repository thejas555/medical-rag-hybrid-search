import os
from groq import Groq


class Generator:

    def __init__(self):

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )


    def build_prompt(self, query, chunks):

        context = ""
        
        for i, chunk in enumerate(chunks):
            context += f"[Source {i+1}] {chunk['text']}\n\n"
            
        prompt = f"""
You are a medical research assistant.

Use ONLY the provided sources to answer the question.

IMPORTANT RULES:
- Do not use outside knowledge.
- Every claim must reference a source using [Source #].
- If the answer cannot be found in the sources, say:
  "The provided documents do not contain sufficient information."

Question:
{query}

Sources:
{context}

Provide a clear answer with source citations like [Source 1].

Answer:
"""

        return prompt


    def generate(self, query, chunks):

        prompt = self.build_prompt(query, chunks)

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content