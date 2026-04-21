from llama_index.llms.ollama import Ollama

llm = Ollama(
    model="phi3",
    request_timeout=120.0,
    context_window=2048   # IMPORTANT FIX
)

response = llm.complete("Explain RAG in one line")

print(response)