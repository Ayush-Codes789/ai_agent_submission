import os
import json
import base64
import operator
from datetime import date
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. SCHEMAS & STATE
# ==========================================
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should be able to do/understand after this section.")
    # REMOVED min_length and max_length constraints to prevent validation crashes
    bullets: List[str] = Field(..., description="3-6 concrete, non-overlapping subpoints to cover in this section.")
    target_words: int = Field(..., description="Target word count for this section (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"

class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str

# ==========================================
# 2. LLM SETUP
# ==========================================
# Primary reasoning model setup (llm1 removed as ChatNVIDIA cannot process SD 3.5 natively)
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0,max_completion_tokens=10000)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     temperature=0
# )# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearch(max_results=max_results)
    try:
        results = tool.invoke({"query": query})
    except Exception as e:
        print(f"Tavily search failed for query '{query}': {e}")
        return []
        
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except json.JSONDecodeError:
            return []

    normalized = []
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict):
                normalized.append({
                    "title": r.get("title") or "No Title",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                })
    return normalized

def _nvidia_generate_image_bytes(prompt: str) -> bytes:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set.")

    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3.5-large"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    
    # Payload adjusted to meet the specific prompt/mode syntax of SD 3.5 Large NIM
    payload = {
        "prompt": prompt,
        "mode": "base",
        "seed": 0,
        "steps": 30
    }

    resp = requests.post(invoke_url, headers=headers, json=payload)
    
    if resp.status_code != 200:
        raise RuntimeError(f"NVIDIA API failed with status {resp.status_code}: {resp.text}")

    response_data = resp.json()
    
    # Robust parsing to catch variations in structural output formats from the NIM catalog
    image_b64 = None
    if "image" in response_data:
        image_b64 = response_data["image"]
    elif "artifacts" in response_data and len(response_data["artifacts"]) > 0:
        image_b64 = response_data["artifacts"][0].get("base64")
    elif "data" in response_data and len(response_data["data"]) > 0:
        image_b64 = response_data["data"][0].get("b64_json")

    if not image_b64:
        raise RuntimeError(f"No base64 image bytes found. Full response structure: {response_data}")

    return base64.b64decode(image_b64)

# ==========================================
# 4. GRAPH NODES
# ==========================================
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.
Decide whether web research is needed BEFORE planning.
Modes:
- closed_book (needs_research=false): Evergreen topics.
- hybrid (needs_research=true): Mostly evergreen but needs up-to-date examples.
- open_book (needs_research=true): Mostly volatile/news.
If needs_research=true, output 3-10 high-signal queries."""

def router_node(state: State) -> dict:
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {topic}"),
    ])
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.
Given raw web search results, produce a deduplicated list of EvidenceItem objects."""

def research_node(state: State) -> dict:
    queries = state.get("queries", []) or []
    raw_results = []
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=6))
    if not raw_results:
        return {"evidence": []}
    
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results:\n{raw_results}"),
    ])
    
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    return {"evidence": list(dedup.values())}

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post (5-9 sections)."""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)
    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")
    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(f"Topic: {state['topic']}\nMode: {mode}\nEvidence:\n{[e.model_dump() for e in evidence][:16]}")),
    ])
    return {"plan": plan}

def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

WORKER_SYSTEM = """You are a senior technical writer.
Write ONE section of a technical blog post in Markdown."""

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence = payload.get("evidence", [])
    evidence_text = "\n".join(f"- {e['title']} | {e['url']}" for e in evidence[:20]) if evidence else ""
    
    section_md = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Topic: {payload['topic']}\nSection title: {task.title}\nGoal: {task.goal}\n"
            f"Target words: {task.target_words}\nBullets:{bullets_text}\n"
            f"Evidence:\n{evidence_text}\n"
        )),
    ]).content.strip()
    return {"sections": [(task.id, section_md)]}

def merge_content(state: State) -> dict:
    plan = state["plan"]
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    return {"merged_md": f"# {plan.blog_title}\n\n{body}\n"}

DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed (max 3). Insert placeholders [[IMAGE_1]], etc."""

def decide_images(state: State) -> dict:
    planner = llm.with_structured_output(GlobalImagePlan)
    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\n\n{state['merged_md']}"),
    ])
    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

def generate_and_place_images(state: State) -> dict:
    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []
    
    if not image_specs:
        return {"final": md}

    images_dir = Path("images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename
        
        if not out_path.exists():
            try:
                img_bytes = _nvidia_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
                img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            except Exception as e:
                img_md = f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n> Error: {e}"
        else:
            img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            
        md = md.replace(placeholder, img_md)

    return {"final": md}

# ==========================================
# 5. ASSEMBLE GRAPH COMPILATION
# ==========================================
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app_graph = g.compile()

# ==========================================
# 6. FASTAPI APPLICATION SETUP
# ==========================================
app = FastAPI(title="AI Blog Agent API")

os.makedirs("images", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")

class GenerateRequest(BaseModel):
    topic: str

@app.get("/")
async def serve_frontend():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    try:
        out = app_graph.invoke({
            "topic": request.topic,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        })
        return {"markdown": out.get("final", "Error: No markdown generated.")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)