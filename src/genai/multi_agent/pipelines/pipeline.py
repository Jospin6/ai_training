from src.genai.multi_agent.agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # search agent working
    print("\n"+"="*50)
    print("step 1 - search is working")
    print("="*50+"\n")

    search_agent = build_search_agent()
    search_results = search_agent.invoke({
        "messages": [
            ("user", f"find recent and reliable information on the topic: {topic}"),
        ]
    })

    state["search_results"] = search_results['messages'][-1].content

    print("\n search results: \n", state["search_results"])

    # step 2 - reader agent working
    print("\n"+"="*50)
    print("step 2 - reader is working")
    print("="*50+"\n")

    reader_agent = build_reader_agent()
    reader_results = reader_agent.invoke({
        "messages": [
            ("user", 
             (f"Based on the following search results about '{topic}'"
             f"pick the most relevant Links (URLs) and scrape it for deeper content.\n\n"
             f"Search Results:\n{state['search_results'][:800]}")

            ),
        ]
    })

    state["scraped_content"] = reader_results['messages'][-1].content

    print("\n scraped content: \n", state["scraped_content"])

    # step 3 - writer agent working
    print("\n"+"="*50)
    print("step 3 - writer is working")
    print("="*50+"\n")

    research_combined = (
        f"Search Results:\n{state['search_results']}\n\n"
        f"Scraped Content:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\n Final Report: \n", state["report"])

    # step 4 - critic agent working
    print("\n"+"="*50)
    print("step 4 - critic is working")
    print("="*50+"\n")

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })
    print("\n Critic Feedback: \n", state["feedback"])


    return state
