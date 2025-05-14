class LLMWrapper:
    def __init__(self, provider: str, **kwargs):
        self.provider = provider
        if provider == "openai":
            from openai import OpenAI
            self.model = OpenAI(**kwargs)
        elif provider == "groq":
            from langchain_groq import ChatGroq
            self.model = ChatGroq(model="mixtral-8x7b-32768")
        else:
            raise ValueError("Unsupported provider")

    def ask(self, prompt: str):
        if self.provider == "openai":
            return self.model.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
        elif self.provider == "groq":
            return self.model.invoke(prompt)

# Usage
if __name__ == "__main__":
    llm = LLMWrapper(provider="openai", api_key="your-key")
    print(llm.ask("What is LangChain?"))
