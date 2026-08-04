from src.genai.multi_agent.tools.tools import web_search, scrape_url


output = web_search("Latest advancements in AI technology")

result = scrape_url.invoke("https://www.futurepedia.io/ai-innovations")

# print(output)
print(result)
    
