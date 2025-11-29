from typing import List, Dict, Any, Optional
import json
import csv

from neo4j import GraphDatabase
from openai import OpenAI


# ---------- Configuration ----------

# OpenAI
OPENAI_API_KEY = ""

# Neo4j AuraDB
NEO4J_URI = ""
NEO4J_USER = ""
NEO4J_PASSWORD = ""

# Existing Neo4j vector index names for ESCO
ESCODISCIPLINE_INDEX = "escodiscipline_embedding_index"
ESCOBOK_INDEX = "escobok_embedding_index"
ESCOKNOWLEDGECONCEPT_INDEX = "escoknowledgeconcept_embedding_index"

# Mapping configuration per RTU node type
# for that node (e.g. "name", "title", "code", etc.).
RTU_TO_ESCO_CONFIG = [
    {
        "rtu_label": "RTUStudyField",
        "rtu_embedding_property": "RTUStudyFieldEmbdng",
        "rtu_mapping_flag_property": "RTUStudyFieldMapped",
        "rtu_label_property": "RTUStudyFieldLabel", 
        "esco_label": "ESCODiscipline",
        "esco_index_name": ESCODISCIPLINE_INDEX,
        "esco_embedding_property": "ESCODisEmbdng",
        "esco_label_property": "ESCODisLabel",
        "relationship_type": "relatesToESCODiscipline",
    },
    {
        "rtu_label": "RTUProgram",
        "rtu_embedding_property": "RTUProgramEmbdng",
        "rtu_mapping_flag_property": "RTUProgramMapped",
        "rtu_label_property": "RTUProgramTitle",
        "esco_label": "ESCOBoK",
        "esco_index_name": ESCOBOK_INDEX,
        "esco_embedding_property": "ESCOBoKEmbdng",
        "esco_label_property": "ESCOBoKLabel",
        "relationship_type": "relatesToESCOBoK",
    },
    {
        "rtu_label": "RTUCourse",
        "rtu_embedding_property": "RTUCourseEmbdng",
        "rtu_mapping_flag_property": "RTUCourseMapped",
        "rtu_label_property": "RTUCourseTitle",
        "esco_label": "ESCOKnowledgeConcept",
        "esco_index_name": ESCOKNOWLEDGECONCEPT_INDEX,
        "esco_embedding_property": "ESCOKnowConEmbdng",
        "esco_label_property": "ESCOKnowCon",
        "relationship_type": "relatesToESCOKnowledgeConcept",
    },
    {
        "rtu_label": "RTUTopic",
        "rtu_embedding_property": "RTUTopicEmbdng",
        "rtu_mapping_flag_property": "RTUTopicMapped",
        "rtu_label_property": "RTUTopicLabel",
        "esco_label": "ESCOKnowledgeConcept",
        "esco_index_name": ESCOKNOWLEDGECONCEPT_INDEX,
        "esco_embedding_property": "ESCOKnowConEmbdng",
        "esco_label_property": "ESCOKnowCon",
        "relationship_type": "relatesToESCOKnowledgeConcept",
    },
]

TOP_K = 5  # number of ESCO candidates to retrieve per RTU node
MODEL_NAME = "gpt-4o"

# ---------- OpenAI LLM helper ----------

def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def decide_mappings_for_rtu(
    client: OpenAI,
    rtu_node: Dict[str, Any],
    rtu_label: str,
    relationship_type: str,
    rtu_embedding_property: str,
    esco_embedding_property: str,
    candidates: List[Dict[str, Any]],
) -> List[str]:
    """
    Use GPT-4o to decide which ESCO candidates should be linked to the RTU node.
    Returns a list of ESCO Neo4j elementIds (strings).
    """
    rtu_id = rtu_node["id"]  # elementId string
    rtu_props = rtu_node["props"]

    # Strip out embeddings from RTU properties
    rtu_clean_props = {
        k: v
        for k, v in rtu_props.items()
        if k != rtu_embedding_property
    }

    cleaned_candidates = []
    for c in candidates:
        props = c["props"]
        clean_props = {
            k: v
            for k, v in props.items()
            if k != esco_embedding_property
        }
        cleaned_candidates.append(
            {
                "neo4j_element_id": c["id"],  # elementId string
                "similarity_score": c["score"],
                "properties": clean_props,
            }
        )

    system_message = {
        "role": "system",
        "content": (
            "You are an expert in curriculum mapping between a university curriculum and "
            "ESCO (European Skills, Competences, Qualifications and Occupations) knowledge elements.\n\n"
            "YOUR TASK\n"
            "From the GIVEN LIST of candidate ESCO concepts, select ALL ESCO knowledge elements "
            "that the given university curriculum element (topic, course, programme, or study field) "
            "could reasonably be expected to teach a typical student.\n\n"
            "OUTPUT FORMAT\n"
            "Return ONLY a single JSON object.\n"
            "Do NOT include any explanations, comments, markdown, or additional text.\n\n"
            "The JSON MUST be valid and follow EXACTLY this schema:\n\n"
            "{\n"
            "  \"rtu_neo4j_element_id\": \"<string>\",\n"
            "  \"selected_esco_ids\": [\"<string>\", \"...\"]\n"
            "}\n\n"
            "Where:\n"
            "- \"rtu_neo4j_element_id\" MUST be exactly the \"element_id\" of the curriculum element from the input.\n"
            "- \"selected_esco_ids\" MUST be a list of unique \"neo4j_element_id\" values taken from the selected "
            "  candidate ESCO concepts.\n"
            "- If none of the candidates can clearly be taught by this curriculum element, return an empty "
            "  list for \"selected_esco_ids\".\n\n"
            "STRICT FORMAT REQUIREMENTS\n"
            "- Use double quotes for all JSON keys and string values.\n"
            "- Do not include trailing commas.\n"
            "- Do not include comments, markdown, or any text before or after the JSON.\n"
        ),
    }

    user_message = {
        "role": "user",
        "content": json.dumps(
            {
                "rtu_node": {
                    "neo4j_element_id": rtu_id,
                    "label": rtu_label,
                    "properties": rtu_clean_props,
                },
                "candidate_esco_nodes": cleaned_candidates,
                "relationship_type": relationship_type,
            },
            ensure_ascii=False,
            indent=2,
        ),
    }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[system_message, user_message],
    )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
        selected_ids_raw = data.get("selected_esco_ids", [])
        if not isinstance(selected_ids_raw, list):
            return []
        # Normalize everything to strings
        selected_ids: List[str] = []
        for x in selected_ids_raw:
            if isinstance(x, str):
                selected_ids.append(x)
            else:
                selected_ids.append(str(x))
        return selected_ids
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for RTU node {rtu_id}: {content}")
        return []


# ---------- Neo4j helpers ----------

def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def fetch_unmapped_rtu_nodes(
    driver,
    rtu_label: str,
    embedding_property: str,
    mapped_flag_property: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Returns RTU nodes that:
      - have an embedding
      - have mapping flag property set to 'no'
    Uses elementId(n) instead of id(n).
    """
    query = f"""
    MATCH (n:{rtu_label})
    WHERE n.{embedding_property} IS NOT NULL
      AND n.{mapped_flag_property} = 'no'
    RETURN elementId(n) AS id, properties(n) AS props
    """
    if limit is not None:
        query += " LIMIT $limit"

    with driver.session() as session:
        result = session.run(query, limit=limit)
        return [{"id": record["id"], "props": record["props"]} for record in result]


def fetch_esco_candidates(
    driver,
    esco_index_name: str,
    esco_label: str,
    embedding_vector: List[float],
    k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Uses Neo4j vector index to get top-k ESCO candidate nodes for a given embedding
    via db.index.vector.queryNodes.
    Uses elementId(node) instead of id(node) to avoid deprecation warnings.
    """
    if embedding_vector is None:
        return []

    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $embedding)
    YIELD node, score
    WHERE $esco_label IN labels(node)
    RETURN elementId(node) AS id, properties(node) AS props, score
    ORDER BY score DESC
    """

    with driver.session() as session:
        result = session.run(
            cypher,
            index_name=esco_index_name,
            k=k,
            embedding=embedding_vector,
            esco_label=esco_label,
        )
        return [
            {"id": record["id"], "props": record["props"], "score": record["score"]}
            for record in result
        ]


def mark_rtu_nodes_mapped(
    driver,
    rtu_label: str,
    mapped_flag_property: str,
    rtu_ids: List[str],
):
    """
    Sets mapped_flag_property = 'yes' on all given RTU nodes.
    Uses elementId(n) instead of id(n).
    """
    if not rtu_ids:
        return

    cypher = f"""
    MATCH (n:{rtu_label})
    WHERE elementId(n) IN $ids
    SET n.{mapped_flag_property} = 'yes'
    """

    with driver.session() as session:
        session.run(cypher, ids=rtu_ids)


def create_mappings(
    driver,
    rtu_label: str,
    esco_label: str,
    relationship_type: str,
    mappings: List[Dict[str, str]],
):
    """
    Create (RTU)-[relationship_type]->(ESCO) relationships for each mapping.
    mappings: list of { 'rtu_id': <elementId string>, 'esco_id': <elementId string> }
    Uses elementId() for matching to avoid id() deprecation.
    """
    if not mappings:
        return

    cypher = f"""
    UNWIND $pairs AS pair
    MATCH (r:{rtu_label}) WHERE elementId(r) = pair.rtu_id
    MATCH (e:{esco_label}) WHERE elementId(e) = pair.esco_id
    MERGE (r)-[rel:`{relationship_type}`]->(e)
    """

    with driver.session() as session:
        session.run(cypher, pairs=mappings)


# ---------- CSV helper ----------

def write_csv_for_rtu_label(rtu_label: str, rows: List[Dict[str, Any]]) -> None:
    """
    Writes a CSV file per RTU label, containing:
      - rtu_neo4j_id       (elementId string)
      - rtu_type
      - rtu_label
      - rag_candidates         (JSON string)
      - gpt_selected_mappings  (JSON string)
    Writes a file only if there is at least one row.
    """
    if not rows:
        return

    filename = f"{rtu_label}_mappings.csv"
    fieldnames = [
        "rtu_neo4j_id",
        "rtu_type",
        "rtu_label",
        "rag_candidates",
        "gpt_selected_mappings",
    ]

    with open(filename, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} mapping rows to {filename}")


# ---------- Main pipeline per config ----------

def run_mapping_pipeline_for_config(
    driver,
    client: OpenAI,
    config: Dict[str, Any],
    limit: Optional[int] = None,
):
    rtu_label = config["rtu_label"]
    rtu_embedding_property = config["rtu_embedding_property"]
    rtu_mapping_flag_property = config["rtu_mapping_flag_property"]
    rtu_label_property = config.get("rtu_label_property")

    esco_label = config["esco_label"]
    esco_index_name = config["esco_index_name"]
    esco_embedding_property = config["esco_embedding_property"]
    esco_label_property = config["esco_label_property"]

    relationship_type = config["relationship_type"]

    print(f"=== Processing {rtu_label} -> {esco_label} ===")

    rtu_nodes = fetch_unmapped_rtu_nodes(
        driver,
        rtu_label=rtu_label,
        embedding_property=rtu_embedding_property,
        mapped_flag_property=rtu_mapping_flag_property,
        limit=limit,
    )

    if not rtu_nodes:
        print(f"No {rtu_label} nodes with {rtu_mapping_flag_property} = 'no' found.")
        return

    all_mappings: List[Dict[str, str]] = []
    considered_ids: List[str] = []
    csv_rows: List[Dict[str, Any]] = []

    for rtu in rtu_nodes:
        rtu_id = rtu["id"]       # elementId string
        rtu_props = rtu["props"]
        embedding_vector = rtu_props.get(rtu_embedding_property)

        if embedding_vector is None:
            print(f"Skipping {rtu_label} {rtu_id} (no embedding).")
            continue

        candidates = fetch_esco_candidates(
            driver,
            esco_index_name=esco_index_name,
            esco_label=esco_label,
            embedding_vector=embedding_vector,
            k=TOP_K,
        )

        considered_ids.append(rtu_id)

        selected_esco_ids = decide_mappings_for_rtu(
            client=client,
            rtu_node=rtu,
            rtu_label=rtu_label,
            relationship_type=relationship_type,
            rtu_embedding_property=rtu_embedding_property,
            esco_embedding_property=esco_embedding_property,
            candidates=candidates,
        )

        # Filter to ESCO elementIds that actually come from the candidate set
        candidate_by_id = {c["id"]: c for c in candidates}
        valid_selected_ids = [
            esco_id for esco_id in selected_esco_ids
            if esco_id in candidate_by_id
        ]

        # Build mappings to write into Neo4j (only for valid selected IDs)
        for esco_id in valid_selected_ids:
            all_mappings.append({"rtu_id": rtu_id, "esco_id": esco_id})

        # ---------- ALWAYS build a CSV row (even if no mappings) ----------

        # RTU label value: use configured property if possible, else JSON of props
        if rtu_label_property and rtu_label_property in rtu_props:
            rtu_label_value = rtu_props[rtu_label_property]
        else:
            # Remove embedding and mapping flag from printed props
            clean_props = {
                k: v for k, v in rtu_props.items()
                if k not in (rtu_embedding_property, rtu_mapping_flag_property)
            }
            rtu_label_value = json.dumps(clean_props, ensure_ascii=False)

        # RAG candidates: label + score
        rag_candidates_list = []
        for c in candidates:
            c_props = c["props"]
            label_value = c_props.get(esco_label_property, "")
            rag_candidates_list.append(
                {
                    "esco_neo4j_id": c["id"],  # elementId string
                    "label": label_value,
                    "score": c["score"],
                }
            )

        # GPT-selected mappings: only those in valid_selected_ids
        # If GPT selected none, this stays an empty list.
        gpt_mappings_list = []
        for esco_id in valid_selected_ids:
            c = candidate_by_id.get(esco_id)
            if not c:
                continue
            c_props = c["props"]
            label_value = c_props.get(esco_label_property, "")
            gpt_mappings_list.append(
                {
                    "esco_neo4j_id": esco_id,
                    "label": label_value,
                    "score": c["score"],
                }
            )

        csv_rows.append(
            {
                "rtu_neo4j_id": rtu_id,
                "rtu_type": rtu_label,
                "rtu_label": rtu_label_value,
                "rag_candidates": json.dumps(rag_candidates_list, ensure_ascii=False),
                "gpt_selected_mappings": json.dumps(gpt_mappings_list, ensure_ascii=False),
            }
        )

    # Write relationships to Neo4j
    if all_mappings:
        print(f"Creating {len(all_mappings)} '{relationship_type}' relationships for {rtu_label}.")
        create_mappings(
            driver,
            rtu_label=rtu_label,
            esco_label=esco_label,
            relationship_type=relationship_type,
            mappings=all_mappings,
        )
    else:
        print(f"No mappings created for {rtu_label} in this run.")

    # Mark all considered RTU nodes as mapped (even if they did not get any relationship)
    if considered_ids:
        print(f"Marking {len(considered_ids)} {rtu_label} nodes as mapped.")
        mark_rtu_nodes_mapped(
            driver,
            rtu_label=rtu_label,
            mapped_flag_property=rtu_mapping_flag_property,
            rtu_ids=considered_ids,
        )

    # Write CSV for this RTU label (rows for every considered node)
    write_csv_for_rtu_label(rtu_label, csv_rows)


def main():
    client = get_openai_client()
    driver = get_driver()

    with driver:
        for config in RTU_TO_ESCO_CONFIG:
            run_mapping_pipeline_for_config(driver, client, config, limit=None)


if __name__ == "__main__":
    main()
