import json
import os
import re
from functools import lru_cache
from typing import AsyncGenerator, Dict, List, TypedDict

import google.auth
from google import genai
from google.genai import types
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel


class AgentState(TypedDict):
    """State passed through the LangGraph workflow."""

    ticker: str
    research: str
    analysis: str
    supervisor_summary: str


class _LLMResponse:
    """Simple wrapper to mimic langchain ChatVertexAI .invoke() return value."""

    def __init__(self, content: str):
        self.content = content


class GenAIGeminiChat:
    """
    Thin wrapper around google.genai Client that provides an .invoke(messages)
    API compatible with the rest of this app.
    """

    def __init__(self, client: genai.Client, model_name: str):
        self.client = client
        self.model_name = model_name

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        parts: List[str] = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            else:
                role = "user"
            parts.append(f"{role.upper()}: {m.content}")
        return "\n\n".join(parts)

    def invoke(self, messages: List[BaseMessage]) -> _LLMResponse:
        prompt = self._messages_to_prompt(messages)

        config = types.GenerateContentConfig(
            max_output_tokens=2048,
            temperature=0.2,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        text = getattr(response, "text", None) or str(response)
        return _LLMResponse(text)


@lru_cache(maxsize=1)
def get_llms_and_tools():
    """Initialize LLMs and tools once, using ADC via google.auth.default()."""
    credentials, project_id = google.auth.default()
    if not project_id:
        raise RuntimeError(
            "No GCP project ID found from default credentials. "
            "Ensure you are running with Application Default Credentials configured."
        )

    # Location "global" is what the working reference uses.
    location = os.getenv("GENAI_LOCATION", "global")
    model_name = os.getenv("VERTEX_MODEL_NAME", "gemini-3.1-pro-preview")

    # This client uses ADC under the hood and matches your reference code.
    client = genai.Client(vertexai=True, project=project_id, location=location)

    supervisor_llm = GenAIGeminiChat(client, model_name)
    researcher_llm = GenAIGeminiChat(client, model_name)
    analyst_llm = GenAIGeminiChat(client, model_name)

    search_tool = DuckDuckGoSearchRun()

    return supervisor_llm, researcher_llm, analyst_llm, search_tool


def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor: orchestrates the flow and keeps a high-level summary."""
    supervisor_llm, _, _, _ = get_llms_and_tools()

    ticker = state["ticker"]
    research = state.get("research", "")
    analysis = state.get("analysis", "")

    print(f"[Supervisor] Orchestrating for ticker={ticker}")

    messages = [
        SystemMessage(
            content=(
                "You are the supervisor of a multi-agent financial intelligence system. "
                "You coordinate a web researcher and a technical risk analyst. "
                "Summarize what has been done so far and what should happen next, "
                "but do not perform research or analysis yourself."
            )
        ),
        HumanMessage(
            content=(
                f"Ticker: {ticker}\n\n"
                f"Current research summary (may be empty):\n{research}\n\n"
                f"Current technical analysis (may be empty):\n{analysis}\n\n"
                "Provide a short supervisor-style narrative (3-5 sentences) "
                "about the workflow status."
            )
        ),
    ]

    response = supervisor_llm.invoke(messages)
    supervisor_summary = response.content if hasattr(response, "content") else str(response)

    print(f"[Supervisor] Summary for {ticker}: {supervisor_summary[:400]}...")

    return {"supervisor_summary": supervisor_summary}


def researcher_node(state: AgentState) -> AgentState:
    """Researcher: uses DuckDuckGo to gather latest information."""
    _, researcher_llm, _, search_tool = get_llms_and_tools()

    ticker = state["ticker"]
    query = f"{ticker} stock latest news, financial performance, and key risks"

    print(f"[Researcher] Running web search for '{query}'")
    raw_results = search_tool.run(query)

    messages = [
        SystemMessage(
            content=(
                "You are a financial research analyst. You receive raw web search "
                "results and must extract the most relevant, recent, and credible "
                "information about a stock, focusing on fundamentals, recent news, "
                "macro context, and notable risks or catalysts."
            )
        ),
        HumanMessage(
            content=(
                f"Ticker: {ticker}\n\n"
                "Here are raw web search results (may be noisy):\n"
                f"{raw_results}\n\n"
                "Summarize the key insights in 5-8 concise bullet points, "
                "emphasizing risk factors, opportunities, and any time-sensitive news."
            )
        ),
    ]

    response = researcher_llm.invoke(messages)
    research_summary = response.content if hasattr(response, "content") else str(response)

    print(f"[Researcher] Summary for {ticker}: {research_summary[:400]}...")

    return {"research": research_summary}


def analyst_node(state: AgentState) -> AgentState:
    """Technical Analyst: converts research into a risk assessment."""
    _, _, analyst_llm, _ = get_llms_and_tools()

    ticker = state["ticker"]
    research = state.get("research", "")

    print(f"[Analyst] Assessing risk for ticker={ticker}")

    messages = [
        SystemMessage(
            content=(
                "You are a technical risk analyst for equities. "
                "Given a research summary, produce a concise risk assessment, "
                "including a qualitative risk level (Low/Medium/High), "
                "a numeric risk score from 0 (no risk) to 100 (extremely risky), "
                "and a short narrative explaining the drivers of risk and opportunity."
            )
        ),
        HumanMessage(
            content=(
                f"Ticker: {ticker}\n\n"
                "Research summary:\n"
                f"{research}\n\n"
                "Respond in the following JSON-like structure (but natural text is fine too):\n"
                "{\n"
                '  \"risk_level\": \"Low | Medium | High\",\n'
                '  \"risk_score\": <0-100>,\n'
                '  \"summary\": \"short narrative\"\n'
                "}\n"
            )
        ),
    ]

    response = analyst_llm.invoke(messages)
    analysis = response.content if hasattr(response, "content") else str(response)

    print(f"[Analyst] Risk assessment for {ticker}: {analysis[:400]}...")

    return {"analysis": analysis}


def supervisor_router(state: AgentState) -> str:
    """Decide the next step based on which parts of the state are populated."""
    # First run: no research yet -> go to researcher
    if not state.get("research"):
        return "researcher"
    # Second run: research exists but no analysis yet -> go to analyst
    if not state.get("analysis"):
        return "analyst"
    # Both research and analysis exist -> end the workflow
    return "end"


@lru_cache(maxsize=1)
def get_graph():
    """Build and cache the LangGraph workflow."""
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("supervisor", supervisor_node)
    graph_builder.add_node("researcher", researcher_node)
    graph_builder.add_node("analyst", analyst_node)

    graph_builder.set_entry_point("supervisor")

    # After each specialist, return to the supervisor for coordination.
    graph_builder.add_edge("researcher", "supervisor")
    graph_builder.add_edge("analyst", "supervisor")

    # Supervisor decides whether to call researcher, analyst, or finish.
    graph_builder.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "end": END,
        },
    )

    graph = graph_builder.compile()
    return graph


class InvokeRequest(BaseModel):
    ticker: str


app = FastAPI(title="Financial Intelligence Multi-Agent Demo")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def stream_graph_updates(ticker: str) -> AsyncGenerator[str, None]:
    """
    Async generator that streams LangGraph execution updates as JSON lines
    formatted as Server-Sent Events (SSE).
    """
    upper_ticker = ticker.upper()

    # Basic ticker format validation: 1–5 uppercase letters.
    if not re.fullmatch(r"[A-Z]{1,5}", upper_ticker):
        error_payload = {
            "event": "error",
            "ticker": upper_ticker,
            "error": "The ticker code is invalid. Please enter a valid stock symbol (e.g. PYPL, AAPL).",
        }
        yield f"data: {json.dumps(error_payload)}\n\n"
        return

    # Semantic validation: ask the model to confirm the ticker exists.
    try:
        supervisor_llm, _, _, _ = get_llms_and_tools()
        validation_messages = [
            SystemMessage(
                content=(
                    "You are a strict stock ticker validation agent. "
                    "Given a proposed ticker, respond with EXACTLY one word: "
                    "'VALID' if it is a real, actively traded stock ticker on a major exchange "
                    "(NYSE, NASDAQ, etc.), otherwise 'INVALID'. "
                    "Do not explain or add anything else."
                )
            ),
            HumanMessage(content=f"Ticker: {upper_ticker}"),
        ]
        validation_response = supervisor_llm.invoke(validation_messages)
        decision = (
            validation_response.content
            if hasattr(validation_response, "content")
            else str(validation_response)
        )
        normalized_decision = (decision or "").strip().upper()
        if not normalized_decision.startswith("VALID"):
            error_payload = {
                "event": "error",
                "ticker": upper_ticker,
                "error": "The ticker code is invalid or not recognized as a real listed stock.",
            }
            yield f"data: {json.dumps(error_payload)}\n\n"
            return
    except Exception as e:
        # If validation itself fails, fall back to running the workflow
        print(f"[API] Ticker validation failed, continuing anyway: {e}")

    graph = get_graph()

    print(f"[API] Streaming /invoke for ticker={upper_ticker}")

    initial_state: AgentState = {
        "ticker": upper_ticker,
        "research": "",
        "analysis": "",
        "supervisor_summary": "",
    }

    try:
        async for event in graph.astream(initial_state, stream_mode="updates"):
            # Each `event` is a mapping of node name -> partial state updates
            for node_name, node_state in event.items():
                if node_name == "__end__":
                    continue

                text_snippet = ""
                if isinstance(node_state, Dict):
                    if "research" in node_state:
                        text_snippet = str(node_state["research"])
                    elif "analysis" in node_state:
                        text_snippet = str(node_state["analysis"])
                    elif "supervisor_summary" in node_state:
                        text_snippet = str(node_state["supervisor_summary"])

                snippet = text_snippet[:400] if text_snippet else ""

                payload = {
                    "node": node_name,
                    "ticker": upper_ticker,
                    "snippet": snippet,
                }

                # SSE format: "data: <json>\n\n"
                yield f"data: {json.dumps(payload)}\n\n"

        # Signal completion explicitly
        done_payload = {"event": "end", "ticker": upper_ticker}
        yield f"data: {json.dumps(done_payload)}\n\n"
    except Exception as e:
        error_payload = {
            "event": "error",
            "ticker": upper_ticker,
            "error": str(e),
        }
        yield f"data: {json.dumps(error_payload)}\n\n"


@app.post("/invoke")
async def invoke(request: InvokeRequest) -> StreamingResponse:
    """
    Invoke the multi-agent financial intelligence workflow for a given ticker,
    streaming node-level updates as Server-Sent Events (SSE).
    """
    return StreamingResponse(
        stream_graph_updates(request.ticker),
        media_type="text/event-stream",
    )


@app.get("/")
async def serve_index():
    """Serve the main UI."""
    return FileResponse("index.html")


@app.get("/analyze")
async def analyze(ticker: str = Query(..., description="Stock ticker symbol, e.g. PYPL")) -> StreamingResponse:
    """
    GET-based streaming endpoint for use with EventSource in the browser.
    """
    return StreamingResponse(
        stream_graph_updates(ticker),
        media_type="text/event-stream",
    )


@app.get("/healthz")
async def healthcheck():
    """Simple health check endpoint for Cloud Run."""
    return {"status": "ok"}

