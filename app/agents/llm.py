from langchain_openai import ChatOpenAI

def build_llm_openrouter(api_key, model, site_url, app_name):
    return ChatOpenAI(model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={"HTTP-Referer": site_url, "X-Title": app_name},
        temperature=0.2)
