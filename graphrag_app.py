"""
GraphRAG Web App — Backend
===========================
Wraps the GraphRAG pipeline in a Flask API.

Usage:
  python graphrag_app.py

Then open http://localhost:5000 in your browser.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle, faiss, numpy as np, json, time, os
from collections import Counter
from openai import OpenAI
from rag_pipeline import load_resources as load_rag_resources, run_rag_query

# ── CONFIG ────────────────────────────────────────────────────────
OPENAI_API_KEY = "insert_key"
GRAPH_PKL  = "Dataset/asrs_graph.pkl"
FAISS_IDX  = "Dataset/faiss_index.bin"
FAISS_META = "Dataset/faiss_metadata.pkl"
MODEL      = "gpt-4o-mini"
TEMP       = 0.2
MAX_TOK    = 900
K          = 5

client = OpenAI(api_key=OPENAI_API_KEY)
app    = Flask(__name__, static_folder="static")
CORS(app)

# ── GLOBALS (loaded once at startup) ──────────────────────────────
G        = None
index    = None
metadata = None

# Load RAG resources
rag_index, rag_metadata = load_rag_resources()

def load_assets():
    global G, index, metadata
    print("Loading assets...")
    with open(GRAPH_PKL, "rb") as f: G = pickle.load(f)
    index = faiss.read_index(FAISS_IDX)
    with open(FAISS_META, "rb") as f: metadata = pickle.load(f)
    print(f"  Graph    : {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"  FAISS    : {index.ntotal:,} vectors")
    print(f"  Metadata : {len(metadata):,} records")
    print("Ready. Open http://localhost:5000")


# ── VOCABULARY ────────────────────────────────────────────────────
VALID_NODES = {
    "PrimaryProblem": [
        "Human Factors", "Aircraft", "Airspace Structure / Procedure", "Airport",
        "ATC Equipment / Nav Facility / Buildings", "Company Policy",
        "Chart Or Publication", "Environment - Non Weather Related",
        "Ground Vehicle / Equipment / Automation", "Manuals", "Staffing",
        "Software and Automation", "Training / Qualification", "Weather", "Procedure", "Other"
    ],
    "ContributingFactor": [
        "Human Factors", "Aircraft", "Airport", "ATC Equipment", "Chart Or Publication",
        "Company Policy", "Environment - Non Weather Related", "Ground Vehicle", "Manuals",
        "Staffing", "Software and Automation", "Training / Qualification", "Weather",
        "Procedure", "Communication Breakdown"
    ],
    "HumanFactor": [
        "Situational Awareness", "Communication Breakdown", "Distraction", "Fatigue",
        "Confusion", "Workload", "Procedure", "Training / Qualification",
        "Physiological - Other", "Psychophysiological", "Attention", "Other"
    ],
    "Anomaly": [
        "Equipment Failure", "Conflict / Near Miss", "Inflight Event",
        "Procedural Deviation", "Ground Event", "ATC Issue", "Passenger / Cabin Event", "Other"
    ],
    "Result": [
        "Crew Intervention", "Emergency / Diversion", "No Action",
        "ATC Intervention", "Aircraft Damage", "Maintenance Action"
    ],
    "FlightPhase": [
        "Cruise", "Taxi", "Parked", "Final Approach", "Initial Approach",
        "Landing", "Climb", "Takeoff / Launch", "Descent", "Initial Climb"
    ],
    "Aircraft": [
        "Boeing 737", "Boeing Widebody", "Airbus Narrowbody", "Airbus Widebody",
        "Regional Jet", "Business Jet", "General Aviation", "Helicopter",
        "UAS / Drone", "Commercial Fixed Wing"
    ],
    "LightCondition": ["Daylight", "Night", "Dusk", "Dawn"],
    "Mission"       : ["Passenger", "Training", "Personal", "Cargo", "Specialized", "UAS", "Other"],
    "Operator"      : ["Air Carrier", "Personal", "Charter / Air Taxi", "Corporate", "UAS", "Government / Military"],
}

TARGET_TYPES = [
    "Result", "ContributingFactor", "HumanFactor", "Anomaly",
    "FlightPhase", "PrimaryProblem", "Aircraft", "LightCondition", "Mission", "Operator"
]

VOCAB_STR = "\n".join(f"  {t}: {', '.join(v)}" for t, v in VALID_NODES.items())

CATCHALL_NODES = {
    "ContributingFactor::Human Factors",
    "ContributingFactor::Aircraft",
    "PrimaryProblem::Human Factors",
}

NODE_TYPE_COLORS = {
    "Anomaly"          : "#f59e0b",
    "Result"           : "#ef4444",
    "HumanFactor"      : "#8b5cf6",
    "ContributingFactor": "#06b6d4",
    "PrimaryProblem"   : "#f97316",
    "FlightPhase"      : "#22c55e",
    "Aircraft"         : "#3b82f6",
    "LightCondition"   : "#eab308",
    "Mission"          : "#ec4899",
    "Operator"         : "#14b8a6",
}


# ── DECOMPOSITION ─────────────────────────────────────────────────
def decompose(query: str) -> dict:
    prompt = (
        'Decompose this aviation safety query for knowledge graph traversal.\n\n'
        'QUERY: "' + query + '"\n\n'
        'VOCABULARY:\n' + VOCAB_STR + '\n\n' +
        '''STEP 1 - SHOULD THIS USE faiss_only?
Return {"mode":"faiss_only"} and STOP if ANY apply:
  a) Negation: "not", "without", "excluding", "non-X", "other than"
  b) Key concept has NO node in vocabulary: go-around, missed approach, CFIT,
     reporter identity, pilot names, airline names, specific airports, tail numbers, dates
  c) "Most vs least probable" within one category -> use single mode instead

STEP 2 - DETECT CONDITIONS (explicit OR implied):
  night/nighttime -> LightCondition::Night | day/daytime -> LightCondition::Daylight
  approach/final approach -> FlightPhase::Final Approach | descent -> FlightPhase::Descent
  helicopter -> Aircraft::Helicopter | Boeing 737 -> Aircraft::Boeing 737
  emergency/diversion -> Result::Emergency / Diversion | near miss/NMAC -> Anomaly::Conflict / Near Miss
  equipment failure -> Anomaly::Equipment Failure | fatigue/tired -> HumanFactor::Fatigue
  communication -> HumanFactor::Communication Breakdown | runway incursion -> Anomaly::Ground Event
  ATC issue -> Anomaly::ATC Issue | crash/accident -> Result::Aircraft Damage
  no action -> Result::No Action | passenger flight -> Mission::Passenger

STEP 3 - CHOOSE MODE:
mode "single"    - one condition, find patterns
  {"mode":"single","anchor_nodes":["NodeType::Value"],"filter_nodes":[],"target_types":["NodeType",...]}
mode "intersect" - two+ different-type conditions that co-occur
  {"mode":"intersect","anchor_nodes":["NodeType::Value"],"filter_nodes":["NodeType::Value"],"target_types":[...]}
mode "compare"   - two named groups to compare: "X vs Y", "differs from", "distinguish X from Y"
  {"mode":"compare","group_a":["NodeType::Value"],"group_b":["NodeType::Value"],"target_types":[...]}

RULES:
1. anchor_nodes: ONE node, ONLY the primary concept. Never add inferred outcome nodes.
2. filter_nodes: ONE node per NodeType max. Each NodeType appears once across anchor+filter.
3. If query mentions only ONE condition, filter_nodes MUST be [].
4. Never put two same-NodeType nodes in anchor or filter - that ANDs them = zero results.
5. target_types: what to DISCOVER - not already in anchor/filter.
   causes -> ContributingFactor, HumanFactor, PrimaryProblem, Anomaly
   outcomes -> Result, Anomaly
   context -> FlightPhase, LightCondition, Aircraft, Mission

Return ONLY the JSON.'''
    )
    resp = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        d   = json.loads(raw)

        def vn(lst):
            out = []
            for n in (lst or []):
                p = n.split("::")
                if len(p) == 2 and p[0] in VALID_NODES and p[1] in VALID_NODES[p[0]]:
                    out.append(n)
            return out

        def vt(lst):
            return [t for t in (lst or []) if t in TARGET_TYPES]

        mode = d.get("mode", "single")
        if mode == "faiss_only":
            return {"mode": "faiss_only"}
        if mode == "compare":
            return {"mode": "compare", "group_a": vn(d.get("group_a", [])),
                    "group_b": vn(d.get("group_b", [])),
                    "target_types": vt(d.get("target_types", TARGET_TYPES[:4]))}

        anchor  = vn(d.get("anchor_nodes", []))
        filters = vn(d.get("filter_nodes", []))
        seen    = {n.split("::")[0] for n in anchor}
        filters = [n for n in filters if n.split("::")[0] not in seen and not seen.add(n.split("::")[0])]
        if not anchor:
            return {"mode": "faiss_only"}
        return {"mode": mode, "anchor_nodes": anchor, "filter_nodes": filters,
                "target_types": vt(d.get("target_types", TARGET_TYPES[:4]))}
    except Exception as e:
        return {"mode": "faiss_only", "error": str(e)}


# ── TRAVERSAL ─────────────────────────────────────────────────────
def incident_set(node: str) -> set:
    if not G.has_node(node): return set()
    return {n for n in G.predecessors(node) if G.nodes[n].get("node_type") == "Incident"}

def incidents_for_group(nodes: list) -> set:
    by_type = {}
    for n in nodes: by_type.setdefault(n.split("::")[0], []).append(n)
    result = None
    for ntype, nlist in by_type.items():
        union = set()
        for n in nlist: union |= incident_set(n)
        result = union if result is None else result & union
    return result or set()

def count_patterns(incidents: set, targets: list, exclude: set = None) -> dict:
    excl = exclude or set()
    pats = {}
    for inc in incidents:
        for _, tgt, _ in G.out_edges(inc, data=True):
            nt = G.nodes[tgt].get("node_type", "")
            lb = G.nodes[tgt].get("label", "")
            if nt not in targets or tgt in excl: continue
            pats.setdefault(nt, Counter())[lb] += 1
    n = len(incidents)
    result = {}
    for nt, ctr in pats.items():
        result[nt] = [
            {"value": v, "count": c, "pct": round(c/n*100, 1),
             "catchall": f"{nt}::{v}" in CATCHALL_NODES}
            for v, c in ctr.most_common(8)
        ]
    return result

def traverse(decomp: dict) -> dict:
    mode = decomp.get("mode")
    if mode == "faiss_only":
        return {"mode": "faiss_only"}

    if mode == "compare":
        ga, gb  = decomp.get("group_a", []), decomp.get("group_b", [])
        targets = decomp.get("target_types", TARGET_TYPES[:4])
        ia = incidents_for_group(ga)
        ib = incidents_for_group(gb)
        if not ia: return {"error": f"Group A empty: {ga}"}
        if not ib: return {"error": f"Group B empty: {gb}"}
        pa = count_patterns(ia, targets, exclude=set(ga))
        pb = count_patterns(ib, targets, exclude=set(gb))
        all_types = set(list(pa.keys()) + list(pb.keys()))
        return {
            "mode": "compare", "group_a": ga, "group_b": gb,
            "group_a_count": len(ia), "group_b_count": len(ib),
            "group_a_pct": round(len(ia)/30513*100, 1),
            "group_b_pct": round(len(ib)/30513*100, 1),
            "comparison": {nt: {"group_a": pa.get(nt,[])[:5], "group_b": pb.get(nt,[])[:5]}
                          for nt in all_types}
        }

    # single / intersect
    anchor  = decomp.get("anchor_nodes", [])
    filters = decomp.get("filter_nodes", [])
    targets = decomp.get("target_types", TARGET_TYPES[:4])

    if not anchor: return {"error": "No anchor nodes"}
    current = incidents_for_group(anchor)
    if not current: return {"error": f"No incidents for {anchor}"}

    breakdown = []
    by_type   = {}
    for fn in filters: by_type.setdefault(fn.split("::")[0], []).append(fn)

    for ntype, fnlist in by_type.items():
        fi = set()
        for fn in fnlist: fi |= incident_set(fn)
        label    = " OR ".join(fn.split("::")[-1] for fn in fnlist)
        narrowed = current & fi
        breakdown.append({
            "filter": label, "before": len(current),
            "after": len(narrowed),
            "pct": round(len(narrowed)/len(current)*100, 1) if current else 0
        })
        if narrowed: current = narrowed
        else: breakdown[-1]["skipped"] = True

    patterns = count_patterns(current, targets, exclude=set(anchor + filters))
    return {
        "mode": mode, "anchor_nodes": anchor, "filter_nodes": filters,
        "matched": len(current), "match_pct": round(len(current)/30513*100, 1),
        "filter_breakdown": breakdown, "patterns": patterns
    }


# ── GENERATION ────────────────────────────────────────────────────
SYS = (
    "You are an aviation safety analyst with two data sources.\n\n"
    "SOURCE 1 - GRAPH TRAVERSAL: Frequency patterns across ALL matched incidents. "
    "Each section is a SEPARATE data dimension. Percentages are within-section only. "
    "HIGH BASE RATE rows appear in most incidents regardless - weak signals.\n\n"
    "SOURCE 2 - SIMILAR INCIDENTS: Specific synopses from semantic search.\n\n"
    "HARD RULES:\n"
    "1. Only cite numbers VERBATIM from SOURCE 1. Never infer sub-breakdowns.\n"
    "2. Never present HIGH BASE RATE nodes as key findings.\n"
    "3. Never sum percentages across different node type sections.\n"
    "4. If faiss_only, limit all claims to narrative evidence only.\n\n"
    "ANSWER STRUCTURE:\n"
    "1. DIRECT ANSWER: 2-3 sentences.\n"
    "2. STATISTICAL PATTERN: SOURCE 1 numbers with node type citations.\n"
    "3. CAUSAL PATHWAY or COMPARISON: Filter narrowing or group differences.\n"
    "4. SPECIFIC EVIDENCE: SOURCE 2 narrative with ACN citations.\n"
    "5. LIMITATIONS: What this data cannot tell you."
)

def build_graph_context(trav: dict) -> str:
    if trav.get("mode") == "faiss_only":
        return ("GRAPH TRAVERSAL: Skipped - concept not in graph vocabulary. "
                "Use FAISS evidence only. No frequency claims.")
    if "error" in trav:
        return f"GRAPH TRAVERSAL ERROR: {trav['error']}. Use FAISS only."
    if trav.get("mode") == "compare":
        la = "+".join(n.split("::")[-1] for n in trav["group_a"])
        lb = "+".join(n.split("::")[-1] for n in trav["group_b"])
        lines = [f"COMPARISON: Group A [{la}]: {trav['group_a_count']:,} incidents | Group B [{lb}]: {trav['group_b_count']:,} incidents"]
        for nt, both in trav["comparison"].items():
            lines.append(f"\n{nt}:")
            lines.append(f"  [{la}]: " + " | ".join(f"{e['value']} {e['pct']}%" for e in both["group_a"][:4]))
            lines.append(f"  [{lb}]: " + " | ".join(f"{e['value']} {e['pct']}%" for e in both["group_b"][:4]))
        return "\n".join(lines)

    lines = [f"TRAVERSAL: Anchor={[n.split('::')[-1] for n in trav['anchor_nodes']]}"]
    if trav.get("filter_breakdown"):
        lines.append("NARROWING: " + " -> ".join(
            f"{s['filter']}:{s['after']:,}" for s in trav["filter_breakdown"]
        ))
    lines.append(f"MATCHED: {trav['matched']:,} of 30,513 ({trav['match_pct']}%)")
    for nt, entries in trav.get("patterns", {}).items():
        lines.append(f"\n{nt}: " + " | ".join(
            f"{e['value']} {e['pct']}%" + (" [CATCH-ALL]" if e["catchall"] else "")
            for e in entries[:5]
        ))
    return "\n".join(lines)

def generate_answer(query, trav, incs):
    gctx  = build_graph_context(trav)
    fctx  = "SIMILAR INCIDENTS:\n" + "\n".join(
        f"[{i+1}] ACN {inc['acn']} | {inc.get('date','')} | "
        f"Problem:{inc.get('primary_problem','')} | "
        f"Result:{inc.get('result','')} | Synopsis:{inc.get('synopsis','')}"
        for i, inc in enumerate(incs)
    )
    resp = client.chat.completions.create(
        model=MODEL, temperature=TEMP, max_tokens=MAX_TOK,
        messages=[
            {"role": "system", "content": SYS},
            {"role": "user", "content": f"QUESTION: {query}\n\nDATA:\n{gctx}\n\n---\n\n{fctx}"}
        ]
    )
    pt = resp.usage.prompt_tokens
    ot = resp.usage.completion_tokens
    return {
        "answer": resp.choices[0].message.content,
        "cost"  : round(pt * 0.00000015 + ot * 0.0000006, 6),
        "tokens": {"prompt": pt, "output": ot}
    }


# ── BUILD GRAPH VIZ DATA ──────────────────────────────────────────
def build_viz_data(decomp: dict, trav: dict) -> dict:
    """
    Build node/edge data for D3 force graph visualization.
    Returns nodes and links the frontend can render.
    """
    nodes = []
    links = []
    node_ids = set()

    def add_node(id_, label, type_, role, value=None, pct=None, catchall=False):
        if id_ not in node_ids:
            node_ids.add(id_)
            nodes.append({
                "id"      : id_,
                "label"   : label,
                "type"    : type_,
                "role"    : role,       # anchor | filter | pattern | query | group_a | group_b
                "value"   : value,
                "pct"     : pct,
                "catchall": catchall,
                "color"   : NODE_TYPE_COLORS.get(type_, "#64748b")
            })

    mode = decomp.get("mode", "single")

    # Query node at center
    add_node("QUERY", "Query", "Query", "query")

    if mode == "faiss_only":
        add_node("FAISS", "FAISS Only\n(no graph match)", "FAISS", "faiss")
        links.append({"source": "QUERY", "target": "FAISS", "type": "query"})
        return {"nodes": nodes, "links": links, "mode": mode}

    if mode == "compare":
        ga = decomp.get("group_a", [])
        gb = decomp.get("group_b", [])

        for n in ga:
            ntype, val = n.split("::")
            add_node(n, val, ntype, "group_a")
            links.append({"source": "QUERY", "target": n, "type": "group_a"})

        for n in gb:
            ntype, val = n.split("::")
            add_node(n, val, ntype, "group_b")
            links.append({"source": "QUERY", "target": n, "type": "group_b"})

        # Pattern nodes for both groups
        for nt, both in trav.get("comparison", {}).items():
            for e in both["group_a"][:4]:
                if e["catchall"]: continue
                nid = f"A_{nt}_{e['value']}"
                add_node(nid, e["value"], nt, "pattern_a", e["value"], e["pct"], e["catchall"])
                src = ga[0] if ga else "QUERY"
                links.append({"source": src, "target": nid, "type": "pattern_a", "pct": e["pct"]})
            for e in both["group_b"][:4]:
                if e["catchall"]: continue
                nid = f"B_{nt}_{e['value']}"
                add_node(nid, e["value"], nt, "pattern_b", e["value"], e["pct"], e["catchall"])
                src = gb[0] if gb else "QUERY"
                links.append({"source": src, "target": nid, "type": "pattern_b", "pct": e["pct"]})

        return {"nodes": nodes, "links": links, "mode": mode,
                "group_a_count": trav.get("group_a_count", 0),
                "group_b_count": trav.get("group_b_count", 0)}

    # single / intersect
    anchors = decomp.get("anchor_nodes", [])
    filters = decomp.get("filter_nodes", [])

    prev = "QUERY"
    for n in anchors:
        ntype, val = n.split("::")
        add_node(n, val, ntype, "anchor")
        links.append({"source": prev, "target": n, "type": "anchor"})
        prev = n

    for i, step in enumerate(trav.get("filter_breakdown", [])):
        fn = filters[i] if i < len(filters) else None
        if fn:
            ntype, val = fn.split("::")
            add_node(fn, f"{val}\n({step['after']:,} incidents)", ntype, "filter")
            links.append({
                "source": prev, "target": fn, "type": "filter",
                "before": step["before"], "after": step["after"], "pct": step["pct"]
            })
            prev = fn

    # Pattern nodes — skip catch-alls for cleaner viz
    for nt, entries in trav.get("patterns", {}).items():
        for e in entries[:5]:
            if e["catchall"]: continue
            nid = f"PAT_{nt}_{e['value']}"
            add_node(nid, e["value"], nt, "pattern", e["value"], e["pct"])
            links.append({"source": prev, "target": nid, "type": "pattern", "pct": e["pct"]})

    return {
        "nodes"  : nodes,
        "links"  : links,
        "mode"   : mode,
        "matched": trav.get("matched", 0),
        "filter_breakdown": trav.get("filter_breakdown", [])
    }


# ── API ROUTES ────────────────────────────────────────────────────
@app.route("/")
def serve_frontend():
    return send_from_directory(".", "graphrag_ui.html")

@app.route("/api/rag-query", methods=["POST"])
def api_rag_query():
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        output = run_rag_query(rag_index, rag_metadata, query)

        incs = []
        for r in output.get("retrieved_results", []):
            incs.append({
                "acn": r.get("acn"),
                "date": str(r.get("date", "")),
                "aircraft_type": str(r.get("aircraft_type", ""))[:60],
                "primary_problem": str(r.get("primary_problem", "")),
                "contributing_factors": str(r.get("contributing_factors", "")),
                "human_factors": str(r.get("human_factors", "")),
                "flight_phase": str(r.get("flight_phase", "")),
                "result": str(r.get("result", "")),
                "synopsis": str(r.get("synopsis", ""))[:500],
                "similarity_score": r.get("similarity_score", 0)
            })

        return jsonify({
            "answer": output.get("answer", ""),
            "incidents": incs,
            "cost_usd": output.get("cost_usd", 0),
            "tokens": {
                "prompt": output.get("prompt_tokens", 0),
                "output": output.get("output_tokens", 0)
            },
            "time_seconds": output.get("time_seconds", 0)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/query", methods=["POST"])
def api_query():
    t0    = time.time()
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    # 1. Decompose
    decomp = decompose(query)

    # 2. Graph traversal
    trav = traverse(decomp)

    # 3. FAISS retrieval
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
    vec  = np.array([resp.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    sc, ix = index.search(vec, K)
    incs = []
    for s, i in zip(sc[0], ix[0]):
        r = metadata[i].copy()
        r["similarity_score"] = round(float(s), 4)
        # Keep only serializable fields
        incs.append({
            "acn"                : r.get("acn"),
            "date"               : str(r.get("date", "")),
            "aircraft_type"      : str(r.get("aircraft_type", ""))[:60],
            "primary_problem"    : str(r.get("primary_problem", "")),
            "contributing_factors": str(r.get("contributing_factors", "")),
            "human_factors"      : str(r.get("human_factors", "")),
            "flight_phase"       : str(r.get("flight_phase", "")),
            "result"             : str(r.get("result", "")),
            "synopsis"           : str(r.get("synopsis", ""))[:500],
            "similarity_score"   : r["similarity_score"],
        })

    # 4. Generate answer
    gen = generate_answer(query, trav, incs)

    # 5. Build viz data
    viz = build_viz_data(decomp, trav)

    return jsonify({
        "query"      : query,
        "answer"     : gen["answer"],
        "decomp"     : decomp,
        "traversal"  : {
            "mode"            : trav.get("mode"),
            "matched"         : trav.get("matched", trav.get("group_a_count", 0)),
            "match_pct"       : trav.get("match_pct", trav.get("group_a_pct", 0)),
            "filter_breakdown": trav.get("filter_breakdown", []),
            "patterns"        : trav.get("patterns", trav.get("comparison", {})),
            "group_a_count"   : trav.get("group_a_count"),
            "group_b_count"   : trav.get("group_b_count"),
        },
        "incidents"  : incs,
        "viz"        : viz,
        "cost_usd"   : gen["cost"],
        "tokens"     : gen["tokens"],
        "time_seconds": round(time.time() - t0, 2),
    })


if __name__ == "__main__":
    load_assets()
    app.run(debug=False, port=5000)
