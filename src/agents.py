import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# Import our tools and LLM setup from Phase 2
from tools import get_llm_with_tools, lookup_policy_docs, web_search_stub, rss_feed_search

# --- 1. Define the State (Enhanced for Visualization) ---
# AgentState is a custom TypedDict that defines the structure of data flowing between nodes.
class AgentState(TypedDict):
    # operator.add ensures we append new messages instead of overwriting existing ones.
    messages: Annotated[List[BaseMessage], operator.add]
    research_data: List[str]
    chart_data: List[dict] # New: Stores structured data for Plotly

# Initialize Resources
llm, llm_with_tools, tools = get_llm_with_tools()

# --- 2. Define the Nodes / Agent Personas ---

def researcher_node(state: AgentState):
    """
    Agent 1: Researcher
    Responsibility: Look up information using tools.
    """
    print("\n--- [Agent: Researcher] is gathering data ---")
    last_message = state["messages"][-1]
    
    # Force the researcher persona via system prompt
    sys_msg = SystemMessage(content="""You are a data gatherer. 
    The current date is February 22, 2026. 
    Use tools to find facts about the user's topic. 
    Do not analyze, just report facts.
    ALWAYS use 'lookup_policy_docs', 'web_search_stub', and 'rss_feed_search' to gather a mix of PDF, Web, and Industry news.""")
    
    # Invoke model
    response = llm_with_tools.invoke([sys_msg, last_message])
    
    research_findings = []
    
    # Execute Tools if requested by the LLM
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"   > Executing Tool: {tool_name}")
            
            # --- ARGUMENT CLEANING LOGIC ---
            # Extract the actual string from the dictionary
            q = tool_args.get('query')
            if isinstance(q, dict): 
                # If nested like {'value': 'search term'} or {'type': 'string', ...}
                q = q.get('value', str(q))
            
            # Fallback for Llama 3.2 schema confusion
            if not q or q == "{'type': 'string'}":
                # Check if the LLM provided more info in the args
                q = tool_args.get('__arg1', tool_args.get('input', 'AI Trends 2026'))
            
            # Convert to string just in case
            q = str(q)
            # -------------------------------

            if tool_name == "lookup_policy_docs":
                res = lookup_policy_docs.invoke(q)
            elif tool_name == "web_search_stub":
                res = web_search_stub.invoke(q)
            elif tool_name == "rss_feed_search":
                res = rss_feed_search.invoke(q)
            
            research_findings.append(f"Source: {tool_name}\nData: {res}")
    else:
        research_findings.append("AGENT: The researcher agent did not call any specific tools for this query. This might happen if the topic is too broad or if the model thinks it has sufficient internal knowledge.")

    print(f"   > Researcher found {len(research_findings)} items.")
    return {
        "messages": [response], 
        "research_data": research_findings
    }

def analyst_node(state: AgentState):
    """
    Agent 2: Analyst (Enhanced with Data Extraction)
    Responsibility: Identify key trends AND extract numeric data for plotting.
    """
    print("\n--- [Agent: Analyst] is identifying trends ---")
    raw_data = "\n\n".join(state["research_data"])
    
    # Note: We use a standard LLM invocation here (no tools bound)
    # because the Analyst only needs to think, not act.
    prompt = f"""You are a senior analyst. 
    1. Identify 3 key trends from the raw data.
    2. DATA VIZ EXTRACTION: Look for REAL numeric trends (percentages, market sizes, years).
       If you find numeric data, extract it into a JSON block like this:
       ```json
       [{{ "label": "2024", "value": 50 }}, {{ "label": "2025", "value": 75 }}]
       ```
    
    CRITICAL: If the raw data is empty or insufficient, DO NOT make up hypothetical numbers. Only extract data that is EXPLICITLY present.
    
    RAW DATA:
    {raw_data}
    """
    
    print(f"   > Analyst node invoking base LLM with {len(raw_data)} chars of raw data...")
    response = llm.invoke(prompt)
    print(f"   > Analyst response received.")
    content = response.content
    
    # Extract JSON if present for Plotly
    chart_data = []
    import json
    import re
    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
    if json_match:
        try:
            chart_data = json.loads(json_match.group(1))
        except:
            pass
            
    return {"messages": [response], "chart_data": chart_data}

def writer_node(state: AgentState):
    """
    Agent 3: Writer (Enhanced for Citations)
    Responsibility: Format analysis into HTML while preserving deep links.
    """
    print("\n--- [Agent: Writer] is formatting the newsletter ---")
    analyst_insight = state["messages"][-1].content
    
    prompt = f"""You are a newsletter editor. 
    Compile the trends into a polite, professional HTML format.
    
    CRITICAL: Preserve all links provided in the analysis (e.g., [Title](URL)).
    Format them as clickable <a> tags in the HTML.
    
    TRENDS & ANALYSIS:
    {analyst_insight}
    """
    
    print(f"   > Writer node invoking base LLM with {len(analyst_insight)} chars of insight...")
    response = llm.invoke(prompt)
    print(f"   > Writer response received.")
    return {"messages": [response]}

# --- 3. Build the Graph ---
# Use StateGraph to define the nodes and how they connect.
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst", analyst_node)
workflow.add_node("Writer", writer_node)

# Add Edges (Define the linear flow)
# Start -> Researcher -> Analyst -> Writer -> End.
workflow.set_entry_point("Researcher")
workflow.add_edge("Researcher", "Analyst")
workflow.add_edge("Analyst", "Writer")
workflow.add_edge("Writer", END)

# Compile the graph
app = workflow.compile()

# --- 4. Runnable Test Block ---
if __name__ == "__main__":
    user_topic = "latest AI trends and internal productivity reports"
    print(f"Starting NewsNexus Agent Team on topic: '{user_topic}'...")
    
    inputs = {"messages": [HumanMessage(content=user_topic)], "research_data": []}
    
    # Run the graph and stream the output
    for output in app.stream(inputs):
        pass # The nodes will print their own status
    
    print("\n\n=== FINAL NEWSLETTER (HTML) ===")
    print(output['Writer']['messages'][-1].content)