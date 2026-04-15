import openai


def generate_answer(query, contexts):
    """Generate answer from retrieved context using OpenAI."""
    context_text = "\n\n".join([c["text"] if isinstance(c, dict) else c for c in contexts])

    prompt = f"""
    You are a helpful AI assistant.

    Answer the question ONLY using the provided context.

    If the context does not contain relevant information, say:
    "I don't have enough information to answer this."

    Do NOT make up answers.

    Context:
    {context_text}

    Question:
    {query}

    Answer clearly and concisely.
    """

    try:
        # Uncomment if using OpenAI API
        # response = openai.ChatCompletion.create(
        #     model="gpt-4o-mini",
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response["choices"][0]["message"]["content"]
        pass
    except Exception as e:
        pass

    # Fallback: return summarized context
    return "Based on retrieved context:\n" + context_text[:500]
