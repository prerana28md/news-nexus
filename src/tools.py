from langchain.tools import tool
@tool
def lookup_policy_docs(query:str)-> str:
    docs=retrive_documents(query,3)

    if not docs:
        return f"No docs found for query: {query}"
    results = []
    for doc ,score in docs:
        source_name=doc.metadata.get("source","unknown pdf")
        basename=os.path.basename(source_name)
        safe_source_path=source_name.replace("\\","/")

        results.append(
            f"(content:{doc.page_content})\n"
            f"Source Link:[{basename}](file://{safe_source_path})")
        return "\n\n".join(results)
@tool
def web_search(query:str)->str:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results=list(ddgs.text(query, max_results=5))
    if not results:
        return "No web results found for query: {query}"
    formatted_results = []
    for res in results:
        formatted_results.append(
            f"Title:{res.get('title')}\n"
            f"Link:[{res.get('title')}]({res.get('href')})\n"
            f"Snippet:{res.get('body')}\n"
        )
    return "\n\n----\n".join(formatted_results)
@tool
def rss_feed_search(query:str)->str:
    import feedparser
    FEEDS=["https://www.technologyreview.com/feed/",
           "https://www.wired.com/feed/rss",
           "https://www.scientificamerican.com/feed/"]
    results=[]
    keywords=query.lower().split()
    for url in FEEDS:
        feed=feedparser.parse(url)
        for entry in feed.entries[:10]:
           (entry.title+""+entry.get("summary","")).lower()
            
           if any(kw in text_to_search for kw in keywords):
                results.append(
                    f"Title:{entry.title}\n"
                    f"Link:[{entry.title}]({entry.link})\n"
                    
                )
def get_llm_with_tools():
    llm=ChatOllama(model="llama3.2",temperature=0)
    tools=[lookup_policy_docs,web_search,rss_feed_search]
    llm_with_tools=llm.bind_tools(tools)
    return llm,llm_with_tools,tools

  
    return "\n\n----\n".join(formatted_entries)