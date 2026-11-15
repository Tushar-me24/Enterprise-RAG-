import logging
from pathlib import Path
from typing import Literal

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None


def llm():
    global _llm
    if _llm is None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        _llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )
    return _llm


def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search


# ---------------------------------------------------------------- helpers

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


def kb_citations(docs):
    seen, out = set(), []
    for d in docs:
        src = Path(str(d.metadata.get("source", "Private KB")).replace("\\", "/")).name
        page = d.metadata.get("page")
        label = f"{src}, page {page + 1}" if isinstance(page, int) else src
        if label not in seen:
            seen.add(label)
            out.append({"title": label, "url": "", "type": "private_kb"})
    return out


def format_sources(citations):
    if not citations:
        return ""
    lines = []
    for i, c in enumerate(citations, 1):
        title, url = c.get("title") or "Source", c.get("url")
        lines.append(f"{i}. [{title}]({url})" if url else f"{i}. {title}")
    return "\n\n---\n**Sources**\n" + "\n".join(lines)


def kb_context(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


# ------------------------------------------------------------------ nodes

def route_question(state: AgentState):
    router = llm().with_structured_output(RouteDecision, method="json_mode")

    prompt = f"""You are a triage router for Nexora Technologies' internal knowledge assistant.
Classify the employee's question into exactly one route.

- "kb": anything about Nexora or its employees' working life: IT and systems (VPN, passwords, MFA,
  email, laptops, software, Wi-Fi, support tickets, data security, onboarding and offboarding),
  HR and leave, working hours, attendance, remote work, policies and procedures. This includes
  questions about the person's OWN account or balance (for example "my leave balance") and any
  question that says "our", "my", "we" or mentions Nexora.
- "external": questions about third-party products, vendors or public information that no company
  policy would contain (for example a Microsoft Teams or Zoom outage, news, general knowledge).
- "direct": ONLY greetings, thanks, goodbyes, compliments or small talk.

If a question could be about Nexora's own policy, choose "kb", never "external".

User question: {state['question']}

Return valid JSON only: {{"route": "kb"}} or {{"route": "external"}} or {{"route": "direct"}}
"""
    decision = router.invoke(prompt)
    return {
        "source_used": decision.route,
        "trace": add_trace(state, f"Router → {decision.route.upper()}"),
    }


def route_after_router(state: AgentState) -> Literal["retrieve_kb", "search_web", "direct_answer"]:
    return {
        "kb": "retrieve_kb",
        "external": "search_web",
        "direct": "direct_answer",
    }.get(state["source_used"], "retrieve_kb")


def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])
    return {
        "kb_docs": docs,
        "trace": add_trace(state, f"Private KB retrieval → {len(docs)} chunks"),
    }


def grade_kb(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    grade = grader.invoke(f"""
You grade evidence for a company knowledge assistant.

Question: {state['question']}

Evidence from the company knowledge base:
{kb_context(state['kb_docs'])}

Return "good" if the evidence contains the specific facts needed to answer the question, even if the
wording differs or the facts are spread across several passages.
If the question asks about a person's own data (for example "my leave balance"), return "good" when the
evidence states the relevant policy or says where employees can check their own data.
Return "weak" only if key facts are missing or the evidence is about something else.

Return valid JSON only: {{"grade": "good"}} or {{"grade": "weak"}}
""")
    return {
        "kb_grade": grade.grade,
        "trace": add_trace(state, f"KB evidence grade → {grade.grade.upper()}"),
    }


def after_kb(state: AgentState) -> Literal["generate_from_kb", "rewrite_query", "insufficient"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"


def rewrite_query(state: AgentState):
    rewritten = llm().invoke(f"""
A search over the company knowledge base did not return enough evidence for this question.
Write ONE alternative search query that keeps the intent but uses different wording: synonyms, the
likely policy or section name, and key terms. Do not answer the question. Return only the query.

Original question: {state['question']}
Previous query: {state['current_query']}
""").content.strip().strip('"')
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }


def generate_from_kb(state: AgentState):
    answer = llm().invoke(f"""
You are the Nexora Technologies company knowledge assistant. Answer ONLY from the private company KB below.

Rules:
- Start with the direct answer, including the exact figure, name or step given in the KB.
- Be concise. Use short numbered steps only when the KB gives steps.
- Never invent details that are not in the KB. If the KB only partly answers the question, say what it
  covers and what it does not.
- If the question asks for personal data you cannot see (for example "my leave balance"), give the
  relevant policy and say where the employee can check their own data if the KB mentions it.
- Do not write a sources list or mention file names; sources are added automatically.

Question: {state['question']}

Private KB:
{kb_context(state['kb_docs'])}
""").content
    citations = kb_citations(state["kb_docs"])
    return {
        "answer": answer + format_sources(citations),
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(state, "Answer generation → PRIVATE KB"),
    }


def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "trace": add_trace(state, "External question → Tavily search"),
    }


def grade_web(state: AgentState):
    grader = llm().with_structured_output(EvidenceGrade, method="json_mode")
    grade = grader.invoke(f"""
Question: {state['question']}

Web evidence:
{state['web_results']}

Return "good" if the evidence is directly relevant and sufficient to answer; otherwise "weak".
Return valid JSON only: {{"grade": "good"}} or {{"grade": "weak"}}
""")
    return {
        "web_grade": grade.grade,
        "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}"),
    }


def after_web(state: AgentState) -> Literal["generate_from_web", "insufficient"]:
    return "generate_from_web" if state["web_grade"] == "good" else "insufficient"


def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are the Nexora Technologies company knowledge assistant. This question is about an external topic,
so it was answered from a web search rather than the private knowledge base.
Answer ONLY from the web evidence below, using only claims the sources directly support.
Start by saying this is external web information that should be verified before changing company systems.
Do not put URLs in the answer; sources are added automatically.

Question: {state['question']}

Web evidence:
{state['web_results']}
""").content
    return {
        "answer": answer + format_sources(state["citations"]),
        "source_used": "web_search",
        "trace": add_trace(state, "Answer generation → WEB SEARCH"),
    }


def direct_answer(state: AgentState):
    answer = llm().invoke(f"Respond briefly and naturally to: {state['question']}").content
    return {
        "answer": answer,
        "source_used": "direct",
        "citations": [],
        "trace": add_trace(state, "Direct response → no retrieval"),
    }


def insufficient(state: AgentState):
    return {
        "answer": (
            "I couldn't find enough reliable evidence in the Nexora knowledge base to answer this "
            "confidently. Please contact the IT Service Desk or HR, or add more detail to your question."
        ),
        "source_used": "insufficient_evidence",
        "citations": [],
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }


# ------------------------------------------------------------------ graph

def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb",
        "search_web": "search_web",
        "direct_answer": "direct_answer",
    })

    # Company questions: KB only. Weak evidence -> rewrite and retry the KB, then refuse.
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb",
        "rewrite_query": "rewrite_query",
        "insufficient": "insufficient",
    })
    graph.add_edge("rewrite_query", "retrieve_kb")

    # External questions: web search only.
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web",
        "insufficient": "insufficient",
    })

    for terminal in ("generate_from_kb", "generate_from_web", "direct_answer", "insufficient"):
        graph.add_edge(terminal, END)
    return graph.compile()


agent_graph = build_graph()


def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)