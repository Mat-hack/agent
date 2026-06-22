from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])

print(
    client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "hello"}]
    )
)