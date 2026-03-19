"""
ASRS GraphRAG Pipeline - Interactive
======================================
Run this script, type any question, get an answer.
Session auto-saves on exit.

Usage:
  python step4_graphrag_pipeline.py

THREE TRAVERSAL MODES:
  single     - one anchor condition, find what patterns emerge
  intersect  - two+ different-type conditions that must co-occur
  compare    - two groups run separately and compared side by side

KEY RULES (enforced in code, not just prompt):
  Filter nodes of SAME type are OR'd together before AND with anchor
  Same-type nodes never AND against each other
  Negation / "not X" / missing vocabulary -> faiss_only
"""

import pickle, faiss, numpy as np, json, time, os
from collections import Counter
from openai import OpenAI

OPENAI_API_KEY = "insert_key"
GRAPH_PKL  = "Dataset/asrs_graph.pkl"
FAISS_IDX  = "Dataset/faiss_index.bin"
FAISS_META = "Dataset/faiss_metadata.pkl"
RESULTS    = "Results"
MODEL      = "gpt-4o-mini"
TEMP       = 0.2
MAX_TOK    = 900
K          = 5

client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs(RESULTS, exist_ok=True)


# == LOAD ======================================================================

def load_assets():
    print("Loading assets...")
    with open(GRAPH_PKL, "rb") as f: G = pickle.load(f)
    index = faiss.read_index(FAISS_IDX)
    with open(FAISS_META, "rb") as f: metadata = pickle.load(f)
    print(f"  Graph    : {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"  FAISS    : {index.ntotal:,} vectors")
    print(f"  Metadata : {len(metadata):,} records")
    return G, index, metadata


# == VOCABULARY ================================================================

VALID_NODES = {
    "PrimaryProblem": [
        "Human Factors", "Aircraft", "Airspace Structure / Procedure", "Airport",
        "ATC Equipment / Nav Facility / Buildings", "Company Policy",
        "Chart Or Publication", "Environment - Non Weather Related",
        "Ground Vehicle / Equipment / Automation", "Manuals", "Staffing",
        "Software and Automation", "Training / Qualification", "Weather",
        "Procedure", "Other"
    ],
    "ContributingFactor": [
        "Human Factors", "Aircraft", "Airport", "ATC Equipment",
        "Chart Or Publication", "Company Policy", "Environment - Non Weather Related",
        "Ground Vehicle", "Manuals", "Staffing", "Software and Automation",
        "Training / Qualification", "Weather", "Procedure", "Communication Breakdown"
    ],
    "HumanFactor": [
        "Situational Awareness", "Communication Breakdown", "Distraction", "Fatigue",
        "Confusion", "Workload", "Procedure", "Training / Qualification",
        "Physiological - Other", "Psychophysiological", "Attention", "Other"
    ],
    "Anomaly": [
        "Equipment Failure", "Conflict / Near Miss", "Inflight Event",
        "Procedural Deviation", "Ground Event", "ATC Issue",
        "Passenger / Cabin Event", "Other"
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
    "Operator"      : ["Air Carrier", "Personal", "Charter / Air Taxi", "Corporate",
                       "UAS", "Government / Military"],
}

TARGET_TYPES = [
    "Result", "ContributingFactor", "HumanFactor", "Anomaly",
    "FlightPhase", "PrimaryProblem", "Aircraft", "LightCondition", "Mission", "Operator"
]

VOCAB_STR = "\n".join(f"  {t}: {', '.join(v)}" for t, v in VALID_NODES.items())

# Catch-all nodes: broad checkbox fields reporters tick in most incidents.
# Flag them so GPT doesn't overweight them as meaningful findings.
CATCHALL_NODES = {
    "ContributingFactor::Human Factors",
    "ContributingFactor::Aircraft",
    "PrimaryProblem::Human Factors",
}

NODE_TYPE_DESCRIPTIONS = {
    "Anomaly"           : "what type of safety event occurred",
    "Result"            : "what outcome or action followed",
    "HumanFactor"       : "specific human performance issues (distinct from PrimaryProblem)",
    "ContributingFactor": "broad contributing categories (Human Factors and Aircraft are catch-alls)",
    "PrimaryProblem"    : "main problem category assigned by reporter",
    "FlightPhase"       : "phase of flight when incident occurred",
    "Aircraft"          : "aircraft category involved",
    "LightCondition"    : "lighting conditions at time of incident",
    "Mission"           : "type of flight operation",
    "Operator"          : "type of operator",
}


# == QUERY DECOMPOSITION =======================================================

def decompose(query: str) -> dict:
    prompt = (
        'Decompose this aviation safety query for knowledge graph traversal.\n\n'
        'QUERY: "' + query + '"\n\n'
        'VOCABULARY (ONLY concepts that exist in the graph):\n'
        + VOCAB_STR + '\n\n'
        + '''
STEP 1 - SHOULD THIS USE faiss_only?
Return {"mode":"faiss_only"} and STOP if ANY of these apply:
  a) Query uses negation: "not", "without", "excluding", "other than", "non-X"
     Example: "incidents without passengers" or "non-passenger flights" -> faiss_only
     Because absence-of-a-category cannot be represented as a graph node
  b) Key concept in the query has NO matching node in the vocabulary above:
     - "go-around" or "missed approach" -> no node exists -> faiss_only
     - "CFIT" or "controlled flight into terrain" -> no node -> faiss_only
     - reporter identity, pilot names, airline names -> no node -> faiss_only
     - specific airports, routes, tail numbers, dates -> no node -> faiss_only
     - specific failure components: autopilot, altimeter, engine number -> no node -> faiss_only
  c) Query asks about "most vs least probable" within a single result category:
     This is a ranking question, NOT a comparison of two groups.
     Use mode=single with the result category as anchor instead.

STEP 2 - DETECT CONDITIONS IN THE QUERY
List every condition the user EXPLICITLY stated or DIRECTLY implied:
  - Flight phase words: approach, climb, descent, landing, takeoff, cruise, taxi
  - Time/light words: night, nighttime, dark, dawn, dusk, daytime
  - Aircraft type: helicopter, Boeing 737, Airbus, drone, UAS, regional jet
  - Outcome words: emergency, crash, damage, diversion, near miss
  - Human factor words: fatigue, distraction, confusion, workload, situational awareness
  - Problem words: equipment failure, weather, procedure, communication, ATC

STEP 3 - CHOOSE ONE MODE:

mode "single" - ONE primary concept, find what patterns surround it
  {"mode":"single","anchor_nodes":["NodeType::Value"],"filter_nodes":[],"target_types":["NodeType",...]}
  anchor_nodes: exactly ONE node representing the primary concept
  filter_nodes: only add if the user stated a second condition of a DIFFERENT NodeType

mode "intersect" - two or more conditions that must co-occur
  Use when query implies multiple conditions even if not using the word AND:
    "nighttime descent"       -> FlightPhase::Descent AND LightCondition::Night
    "helicopter emergencies"  -> Aircraft::Helicopter AND Result::Emergency / Diversion
    "fatigue during approach" -> HumanFactor::Fatigue AND FlightPhase::Final Approach
  {"mode":"intersect","anchor_nodes":["NodeType::Value"],"filter_nodes":["NodeType::Value"],"target_types":["NodeType",...]}
  anchor_nodes: primary condition (1 node)
  filter_nodes: secondary condition(s), MUST be DIFFERENT NodeType from anchor AND from each other

mode "compare" - two explicitly named groups to compare side by side
  Use for: "X vs Y", "differs from", "compared to", "distinguish X from Y", "between X and Y"
  This includes: "distinguish incidents with aircraft damage from those with no action taken"
  {"mode":"compare","group_a":["NodeType::Value"],"group_b":["NodeType::Value"],"target_types":["NodeType",...]}
  group_a and group_b should each have 1 node of the same NodeType

PLAIN-ENGLISH TO NODE MAPPINGS (use these exact values):
  crash or accident           -> Result::Aircraft Damage
  emergency or diversion      -> Result::Emergency / Diversion
  no action taken             -> Result::No Action
  crew intervened             -> Result::Crew Intervention
  near miss or NMAC           -> Anomaly::Conflict / Near Miss
  equipment failure           -> Anomaly::Equipment Failure
  procedural deviation        -> Anomaly::Procedural Deviation
  runway incursion            -> Anomaly::Ground Event
  ATC issue                   -> Anomaly::ATC Issue
  fatigue or tired            -> HumanFactor::Fatigue
  communication issues        -> HumanFactor::Communication Breakdown
  situational awareness       -> HumanFactor::Situational Awareness
  distraction                 -> HumanFactor::Distraction
  confusion                   -> HumanFactor::Confusion
  workload                    -> HumanFactor::Workload
  night or nighttime          -> LightCondition::Night
  day or daytime              -> LightCondition::Daylight
  weather                     -> ContributingFactor::Weather
  software or automation      -> PrimaryProblem::Software and Automation
  helicopter                  -> Aircraft::Helicopter
  drone or UAS                -> Aircraft::UAS / Drone
  Boeing 737                  -> Aircraft::Boeing 737
  airline or air carrier      -> Operator::Air Carrier
  passenger flight            -> Mission::Passenger

CRITICAL RULES - READ CAREFULLY:
1. anchor_nodes: exactly ONE node, representing ONLY the primary concept in the query.
   NEVER add outcome/result nodes to anchor unless the query is specifically asking about that outcome.
   WRONG: "issues in final approach" -> anchor=["FlightPhase::Final Approach","Result::Crew Intervention"]
   RIGHT: "issues in final approach" -> anchor=["FlightPhase::Final Approach"]

2. filter_nodes: MAXIMUM ONE node per NodeType.
   If query mentions multiple human factors (e.g. fatigue, distraction, confusion):
     - If asking what co-occurs WITH ONE of them -> anchor on that one factor, target_types=["HumanFactor",...]
     - Do NOT list all human factors as filter_nodes - that ANDs them and returns near zero results
   WRONG: filter_nodes=["HumanFactor::Fatigue","HumanFactor::Distraction","HumanFactor::Confusion"]
   RIGHT: anchor_nodes=["HumanFactor::Fatigue"], filter_nodes=[], target_types=["HumanFactor","Result",...]

3. filter_nodes NodeType must differ from anchor_nodes NodeType AND from each other.
   Each NodeType can appear at most once across anchor_nodes + filter_nodes combined.
   If the query mentions only ONE condition, filter_nodes MUST be empty [].
   WRONG: query="nighttime travel" -> filter_nodes=["FlightPhase::Cruise"] (user never mentioned cruise)
   RIGHT: query="nighttime travel" -> anchor=["LightCondition::Night"], filter_nodes=[]

4. target_types: what we want to DISCOVER. Do NOT include NodeTypes already in anchor/filter.
   cause queries    -> ContributingFactor, HumanFactor, PrimaryProblem, Anomaly
   outcome queries  -> Result, Anomaly
   context queries  -> FlightPhase, LightCondition, Aircraft, Mission

5. If a key concept has no exact match in vocabulary -> faiss_only. Do not approximate with an unrelated node.

Return ONLY the JSON, no explanation.'''
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
            return {
                "mode"        : "compare",
                "group_a"     : vn(d.get("group_a", [])),
                "group_b"     : vn(d.get("group_b", [])),
                "target_types": vt(d.get("target_types", TARGET_TYPES[:4]))
            }

        anchor  = vn(d.get("anchor_nodes", []))
        filters = vn(d.get("filter_nodes", []))

        # SAFETY: enforce one node per NodeType across anchor + filter combined
        # Remove any filter node whose type already appears in anchor or earlier filter
        seen_types = {n.split("::")[0] for n in anchor}
        clean_filters = []
        for n in filters:
            ntype = n.split("::")[0]
            if ntype not in seen_types:
                clean_filters.append(n)
                seen_types.add(ntype)
            # silently drop same-type duplicates

        if not anchor:
            return {"mode": "faiss_only"}

        return {
            "mode"        : mode,
            "anchor_nodes": anchor,
            "filter_nodes": clean_filters,
            "target_types": vt(d.get("target_types", TARGET_TYPES[:4]))
        }
    except Exception as e:
        return {"mode": "faiss_only", "error": str(e)}


# == TRAVERSAL =================================================================

def incident_set(G, node: str) -> set:
    if not G.has_node(node):
        return set()
    return {n for n in G.predecessors(node) if G.nodes[n].get("node_type") == "Incident"}


def incidents_for_group(G, nodes: list) -> set:
    """
    OR within same NodeType, AND across different NodeTypes.
    This is the core rule that prevents same-type AND returning zero results.

    Example: ["HumanFactor::Fatigue", "HumanFactor::Distraction", "LightCondition::Night"]
    -> (Fatigue OR Distraction) AND Night
    NOT Fatigue AND Distraction AND Night (which returns near zero)
    """
    by_type = {}
    for n in nodes:
        by_type.setdefault(n.split("::")[0], []).append(n)
    result = None
    for ntype, nlist in by_type.items():
        union = set()
        for n in nlist:
            union |= incident_set(G, n)
        result = union if result is None else result & union
    return result or set()


def count_patterns(G, incidents: set, targets: list, exclude: set = None) -> dict:
    excl = exclude or set()
    pats = {}
    for inc in incidents:
        for _, tgt, _ in G.out_edges(inc, data=True):
            nt = G.nodes[tgt].get("node_type", "")
            lb = G.nodes[tgt].get("label", "")
            if nt not in targets or tgt in excl:
                continue
            pats.setdefault(nt, Counter())[lb] += 1
    n = len(incidents)
    result = {}
    for nt, ctr in pats.items():
        entries = []
        for v, c in ctr.most_common(8):
            entries.append({
                "value"   : v,
                "count"   : c,
                "pct"     : round(c / n * 100, 1),
                "catchall": f"{nt}::{v}" in CATCHALL_NODES,
            })
        result[nt] = entries
    return result


def traverse_single(G, decomp: dict) -> dict:
    anchor  = decomp.get("anchor_nodes", [])
    filters = decomp.get("filter_nodes", [])
    targets = decomp.get("target_types", TARGET_TYPES[:4])

    if not anchor:
        return {"error": "No valid anchor nodes"}

    current = incidents_for_group(G, anchor)
    if not current:
        return {"error": f"No incidents found for: {anchor}"}

    # Apply filters: group by type first (OR within type), then AND across types
    # This prevents same-type filter nodes from returning zero results
    breakdown = []
    by_type   = {}
    for fn in filters:
        by_type.setdefault(fn.split("::")[0], []).append(fn)

    for ntype, fnlist in by_type.items():
        # Union of same-type filter nodes
        fi = set()
        for fn in fnlist:
            fi |= incident_set(G, fn)
        label    = " OR ".join(fn.split("::")[-1] for fn in fnlist)
        narrowed = current & fi
        breakdown.append({
            "filter" : label,
            "before" : len(current),
            "after"  : len(narrowed),
            "pct"    : round(len(narrowed) / len(current) * 100, 1) if current else 0,
        })
        if narrowed:
            current = narrowed
        else:
            breakdown[-1]["skipped"] = True

    patterns = count_patterns(G, current, targets, exclude=set(anchor + filters))
    return {
        "mode"            : decomp["mode"],
        "anchor_nodes"    : anchor,
        "filter_nodes"    : filters,
        "matched"         : len(current),
        "match_pct"       : round(len(current) / 30513 * 100, 1),
        "filter_breakdown": breakdown,
        "patterns"        : patterns,
    }


def traverse_compare(G, decomp: dict) -> dict:
    ga, gb  = decomp.get("group_a", []), decomp.get("group_b", [])
    targets = decomp.get("target_types", TARGET_TYPES[:4])

    if not ga or not gb:
        return {"error": "Compare mode requires two non-empty groups"}

    ia = incidents_for_group(G, ga)
    ib = incidents_for_group(G, gb)
    if not ia: return {"error": f"Group A returned no incidents: {ga}"}
    if not ib: return {"error": f"Group B returned no incidents: {gb}"}

    pa = count_patterns(G, ia, targets, exclude=set(ga))
    pb = count_patterns(G, ib, targets, exclude=set(gb))
    all_types  = set(list(pa.keys()) + list(pb.keys()))
    comparison = {
        nt: {"group_a": pa.get(nt, [])[:5], "group_b": pb.get(nt, [])[:5]}
        for nt in all_types
    }

    return {
        "mode"          : "compare",
        "group_a"       : ga,
        "group_b"       : gb,
        "group_a_count" : len(ia),
        "group_b_count" : len(ib),
        "group_a_pct"   : round(len(ia) / 30513 * 100, 1),
        "group_b_pct"   : round(len(ib) / 30513 * 100, 1),
        "comparison"    : comparison,
    }


# == FORMAT CONTEXT ============================================================

def fmt_traversal(t: dict) -> str:
    if t.get("mode") == "faiss_only":
        return (
            "GRAPH TRAVERSAL: Skipped.\n"
            "This query involves a concept not represented as a node in the graph "
            "(e.g. negation of a category, go-around, missed approach, specific location, "
            "reporter identity, or a concept with no vocabulary match).\n"
            "Answer is based on FAISS semantic search only. "
            "Do NOT make frequency claims across the full dataset."
        )
    if "error" in t:
        return f"GRAPH TRAVERSAL ERROR: {t['error']}\nFalling back to FAISS only."
    if t.get("mode") == "compare":
        return fmt_compare(t)
    return fmt_single(t)


def fmt_single(t: dict) -> str:
    lines = ["GRAPH TRAVERSAL - INTERSECTION ANALYSIS"]
    lines.append(f"Anchor   : {' + '.join(n.split('::')[-1] for n in t['anchor_nodes'])}")
    if t["filter_nodes"]:
        lines.append(f"Filters  : {' + '.join(n.split('::')[-1] for n in t['filter_nodes'])}")
    lines.append(f"Matched  : {t['matched']:,} of 30,513 total incidents ({t['match_pct']}%)")

    if t["filter_breakdown"]:
        lines.append("\nFILTER NARROWING (how conditions stack):")
        for s in t["filter_breakdown"]:
            sk = " [skipped - would empty results]" if s.get("skipped") else ""
            lines.append(
                f"  + {s['filter']}: {s['after']:,} incidents "
                f"({s['pct']}% of previous){sk}"
            )

    lines.append(f"\nPATTERNS ACROSS {t['matched']:,} MATCHED INCIDENTS:")
    lines.append("NOTE: Each section is a SEPARATE data dimension.")
    lines.append("      Percentages are within-section only. Do NOT add across sections.")

    for nt, entries in t["patterns"].items():
        desc = NODE_TYPE_DESCRIPTIONS.get(nt, "")
        lines.append(f"\n  {nt} - {desc}:")
        for e in entries:
            flag = "  [HIGH BASE RATE - appears in most incidents, weak signal]" if e["catchall"] else ""
            lines.append(f"    {e['count']:>5} ({e['pct']:>5}%)  {e['value']}{flag}")

    return "\n".join(lines)


def fmt_compare(t: dict) -> str:
    la = " + ".join(n.split("::")[-1] for n in t["group_a"])
    lb = " + ".join(n.split("::")[-1] for n in t["group_b"])
    lines = ["GRAPH TRAVERSAL - COMPARISON ANALYSIS"]
    lines.append(f"Group A [{la}]: {t['group_a_count']:,} incidents ({t['group_a_pct']}%)")
    lines.append(f"Group B [{lb}]: {t['group_b_count']:,} incidents ({t['group_b_pct']}%)")
    lines.append("NOTE: Compare same row across Group A vs B to find real differences.")
    lines.append("      Percentages are within-group. Do NOT add across groups or sections.")

    for nt, both in t["comparison"].items():
        desc = NODE_TYPE_DESCRIPTIONS.get(nt, "")
        lines.append(f"\n  {nt} - {desc}:")
        lines.append(f"    Group A [{la}]:")
        for e in both["group_a"]:
            flag = "  [HIGH BASE RATE]" if e["catchall"] else ""
            lines.append(f"      {e['count']:>5} ({e['pct']:>5}%)  {e['value']}{flag}")
        lines.append(f"    Group B [{lb}]:")
        for e in both["group_b"]:
            flag = "  [HIGH BASE RATE]" if e["catchall"] else ""
            lines.append(f"      {e['count']:>5} ({e['pct']:>5}%)  {e['value']}{flag}")

    return "\n".join(lines)


def fmt_faiss(incidents: list) -> str:
    lines = ["SEMANTICALLY SIMILAR INCIDENTS (FAISS):"]
    for i, inc in enumerate(incidents, 1):
        lines.append(
            f"\n[{i}] ACN {inc['acn']} | Score {inc['similarity_score']} | "
            f"{inc.get('date', '')} | {str(inc.get('aircraft_type', ''))[:40]}"
        )
        lines.append(f"    Primary Problem : {inc.get('primary_problem', '')}")
        lines.append(f"    Contributing    : {inc.get('contributing_factors', '')}")
        lines.append(f"    Human Factors   : {inc.get('human_factors', '')}")
        lines.append(f"    Flight Phase    : {inc.get('flight_phase', '')}")
        lines.append(f"    Result          : {inc.get('result', '')}")
        lines.append(f"    Synopsis        : {inc.get('synopsis', '')}")
    return "\n".join(lines)


# == GENERATION ================================================================

SYS = (
    "You are an aviation safety analyst with two data sources.\n\n"
    "SOURCE 1 - GRAPH TRAVERSAL\n"
    "Frequency patterns across ALL matched incidents in the dataset.\n"
    "Each section is a SEPARATE data dimension - percentages within each section "
    "show share of matched incidents that have that attribute.\n"
    "Rows marked HIGH BASE RATE appear in most incidents regardless of query - "
    "they are weak signals and should NOT be presented as key findings.\n\n"
    "SOURCE 2 - SIMILAR INCIDENTS\n"
    "Specific incident synopses from semantic search. Use for narrative detail and ACN citations.\n\n"
    "HARD RULES - NEVER VIOLATE:\n"
    "1. Only cite numbers that appear VERBATIM in SOURCE 1. "
    "Never compute or infer sub-breakdowns not explicitly listed.\n"
    "   If a figure is not in SOURCE 1, do not state it - acknowledge the gap instead.\n"
    "2. Never cross-reference SOURCE 1 and SOURCE 2 numbers to derive new statistics.\n"
    "3. Never present HIGH BASE RATE nodes as key findings. "
    "Always note they are common across all incidents.\n"
    "4. Sections in SOURCE 1 measure different things. Never sum percentages across sections.\n"
    "   ContributingFactor and PrimaryProblem are different schema fields, not the same dimension.\n"
    "5. If graph traversal was skipped (faiss_only), say so and limit all claims to narrative evidence only.\n\n"
    "ANSWER STRUCTURE:\n"
    "1. DIRECT ANSWER: 2-3 sentences directly answering the question.\n"
    "2. STATISTICAL PATTERN: Numbers from SOURCE 1 - cite the node type for each number.\n"
    "3. CAUSAL PATHWAY or COMPARISON: Filter narrowing (if present) or group differences.\n"
    "4. SPECIFIC EVIDENCE: Narrative details from SOURCE 2 with ACN citations.\n"
    "5. LIMITATIONS: What this data cannot tell you, including HIGH BASE RATE caveats."
)


def generate(query: str, gctx: str, fctx: str) -> dict:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL, temperature=TEMP, max_tokens=MAX_TOK,
        messages=[
            {"role": "system", "content": SYS},
            {"role": "user",
             "content": f"QUESTION: {query}\n\nDATA:\n{gctx}\n\n{'='*60}\n\n{fctx}"}
        ]
    )
    pt   = resp.usage.prompt_tokens
    ot   = resp.usage.completion_tokens
    cost = round(pt * 0.00000015 + ot * 0.0000006, 6)
    return {
        "answer": resp.choices[0].message.content,
        "time"  : round(time.time() - t0, 2),
        "pt": pt, "ot": ot, "cost": cost,
    }


# == MAIN QUERY FUNCTION =======================================================

def run_query(query: str, G, index, metadata, verbose: bool = True) -> dict:
    t0 = time.time()

    if verbose: print("  [1/4] Decomposing...", end=" ", flush=True)
    decomp = decompose(query)
    if verbose: print(f"mode={decomp['mode']}")

    if verbose: print("  [2/4] Graph traversal...", end=" ", flush=True)
    if decomp["mode"] == "faiss_only":
        trav    = {"mode": "faiss_only", "matched": 0, "match_pct": 0}
        matched = 0
    elif decomp["mode"] == "compare":
        trav    = traverse_compare(G, decomp)
        matched = trav.get("group_a_count", 0)
    else:
        trav    = traverse_single(G, decomp)
        matched = trav.get("matched", 0)
    if verbose: print(f"{matched:,} matched")

    if verbose: print("  [3/4] FAISS retrieval...", end=" ", flush=True)
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
    vec  = np.array([resp.data[0].embedding], dtype=np.float32)
    faiss.normalize_L2(vec)
    sc, ix = index.search(vec, K)
    incs = []
    for s, i in zip(sc[0], ix[0]):
        r = metadata[i].copy()
        r["similarity_score"] = round(float(s), 4)
        incs.append(r)
    if verbose: print(f"top-{len(incs)}")

    if verbose: print("  [4/4] Generating...", end=" ", flush=True)
    gctx = fmt_traversal(trav)
    fctx = fmt_faiss(incs)
    gen  = generate(query, gctx, fctx)
    if verbose: print(f"{gen['time']}s  ${gen['cost']}")

    return {
        "query"            : query,
        "pipeline"         : "GraphRAG",
        "answer"           : gen["answer"],
        "decomposition"    : decomp,
        "traversal"        : trav,
        "retrieved_acns"   : [i["acn"] for i in incs],
        "similarity_scores": [i["similarity_score"] for i in incs],
        "time_seconds"     : round(time.time() - t0, 2),
        "prompt_tokens"    : gen["pt"],
        "output_tokens"    : gen["ot"],
        "cost_usd"         : gen["cost"],
    }


# == INTERACTIVE ===============================================================

def interactive(G, index, metadata):
    session    = []
    total_cost = 0.0
    fname      = os.path.join(RESULTS, f"graphrag_session_{int(time.time())}.json")

    print("\n" + "=" * 60)
    print("GraphRAG - Interactive Mode")
    print("Ask any question about aviation safety incidents.")
    print("Commands: quit | save | cost")
    print("=" * 60)

    while True:
        print()
        try:
            q = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q: continue
        if q.lower() == "quit": break
        if q.lower() == "cost":
            print(f"Session: {len(session)} queries, ${total_cost:.4f}")
            continue
        if q.lower() == "save":
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2)
            print(f"Saved to {fname}")
            continue

        result = run_query(q, G, index, metadata, verbose=True)
        total_cost += result["cost_usd"]
        session.append(result)

        print()
        print("-" * 60)
        print(result["answer"])
        print("-" * 60)

        d = result["decomposition"]
        t = result["traversal"]
        print(f"\nMode    : {d['mode']}")
        if d["mode"] == "compare":
            print(f"Group A : {d.get('group_a')}")
            print(f"Group B : {d.get('group_b')}")
        elif d["mode"] != "faiss_only":
            print(f"Anchors : {d.get('anchor_nodes')}")
            print(f"Filters : {d.get('filter_nodes')}")
        print(f"Matched : {t.get('matched', t.get('group_a_count', 0)):,}")
        print(f"ACNs    : {result['retrieved_acns']}")
        print(f"Time    : {result['time_seconds']}s  "
              f"Cost: ${result['cost_usd']}  "
              f"Session: ${total_cost:.4f}")

    if session:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)
        print(f"\nSession saved: {fname}")
        print(f"Queries: {len(session)}  Total cost: ${total_cost:.4f}")


# == ENTRY POINT ===============================================================

if __name__ == "__main__":
    G, index, metadata = load_assets()
    interactive(G, index, metadata)
