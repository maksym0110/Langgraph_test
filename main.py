from dotenv import load_dotenv

load_dotenv()

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# ============================================================
# STATE
# ============================================================
# State = the memory that moves through the graph.
# Here we store the chat messages.
# add_messages means new messages get appended to history.
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# TOOL 1: PostgreSQL
# ============================================================
# Use this when the agent needs structured data:
# - orders
# - invoices
# - customers
# - transactions
# ============================================================
@tool
def query_postgres(sql: str) -> str:
    """Run a SQL query against PostgreSQL."""
    # Fake result for learning purpose
    return f"PostgreSQL result for query: {sql}"


# ============================================================
# TOOL 2: Vector DB
# ============================================================
# Use this when the agent needs semantic search:
# - FAQ
# - policies
# - help docs
# - knowledge base
# ============================================================
@tool
def search_vector_db(query: str) -> str:
    """Search knowledge base in a vector database."""
    return f"Vector DB matches for: {query}"


# ============================================================
# TOOL 3: CRM
# ============================================================
# Use this when the agent needs customer relationship data:
# - customer profile
# - account notes
# - lead status
# - follow-up actions
# ============================================================
@tool
def update_crm(customer_id: str, note: str) -> str:
    """Update a CRM record with a note."""
    return f"CRM updated for customer {customer_id}: {note}"


# ============================================================
# TOOL 4: APIs
# ============================================================
# Use this for external services:
# - shipping API
# - payment API
# - ticketing API
# - weather API
# ============================================================
@tool
def call_shipping_api(order_id: str) -> str:
    """Check shipping status from an external API."""
    fake_status = {
        "A100": "delivered",
        "A101": "shipped",
        "A102": "delayed"
    }
    status = fake_status.get(order_id, "not found")
    return f"Shipping status for {order_id}: {status}"


# ============================================================
# TOOL 5: Documents
# ============================================================
# Use this when the agent needs to read files:
# - invoices
# - PDFs
# - contracts
# - SOPs
# - manuals
# ============================================================
@tool
def read_document(file_name: str) -> str:
    """Read a business document."""
    return f"Document content from {file_name}"


tools = [
    query_postgres,
    search_vector_db,
    update_crm,
    call_shipping_api,
    read_document
]


# ============================================================
# LLM
# ============================================================
# OpenAI model with tools bound to it.
# That means the model can decide when to call tools.
# ============================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


# ============================================================
# PROMPT
# ============================================================
# This instruction controls the single agent behavior.
# ============================================================
SYSTEM_PROMPT = SystemMessage(content=
    "You are a single customer operations agent. "
    "Use PostgreSQL for structured business data. "
    "Use the vector database for policies and knowledge base lookup. "
    "Use CRM for account notes and customer updates. "
    "Use external APIs for live service status. "
    "Use documents when the answer is stored in files. "
    "Be concise, accurate, and practical."
)


# ============================================================
# AGENT NODE
# ============================================================
# Node = one step in the graph.
# This node sends the current state to the LLM.
# The LLM may answer directly or request a tool call.
# ============================================================
def agent_node(state: AgentState):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ============================================================
# BUILD GRAPH
# ============================================================
graph = StateGraph(AgentState)

# Add the agent node.
graph.add_node("agent", agent_node)

# Add the tool execution node.
graph.add_node("tools", ToolNode(tools))

# Set the first node.
graph.set_entry_point("agent")


# ============================================================
# EDGES
# ============================================================
# Edge = path from one node to another.
#
# tools_condition does this:
# - if the LLM asked for a tool -> go to tools
# - otherwise -> stop
#
# Then after tools run, go back to the agent so it can read the result.
# ============================================================
graph.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",
        END: END
    }
)

graph.add_edge("tools", "agent")


# Compile the graph into a runnable app.
app = graph.compile()


# ============================================================
# VISUALIZATION
# ============================================================
def save_graph_visualization(output_dir: str = ".") -> None:
    """Save LangGraph workflow diagram as Mermaid source and PNG."""
    graph_viz = app.get_graph()

    mmd_path = f"{output_dir}/agent_graph.mmd"
    png_path = f"{output_dir}/agent_graph.png"

    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(graph_viz.draw_mermaid())

    with open(png_path, "wb") as f:
        f.write(graph_viz.draw_mermaid_png())




# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    result = app.invoke({
        "messages": [
            HumanMessage(
                content="Check order A100, see the policy for delayed shipments, and update the CRM note."
            )
        ]
    })

    for msg in result["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content}")

    save_graph_visualization()