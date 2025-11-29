"""
Create ESCO GraphRAG schema in Neo4j AuraDB from CSVs,
including an additional ESCOKnowledgeGroup label and
a vector index escoknowledgegroup_embedding_index
for ESCOKnowledgeConcept nodes that have outgoing
:consistsOfKnowledgeUnit edges.

Requirements:
    pip install pandas openai neo4j
"""

import os
from typing import List, Dict, Iterable

import pandas as pd
from openai import OpenAI
from neo4j import GraphDatabase


# =========================
# CONFIGURATION
# =========================

# --- File paths ---
KNOWLEDGE_HIERARCHY_CSV_PATH = "" #Path to ESCO Hierarchy
KNOWLEDGE_MAPPING_CSV_PATH = "" #Path to ESCO Knowledge Group to Knowledge Concept Mappings

# --- OpenAI ---
OPENAI_API_KEY = ""
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 64  # tune if needed

# --- Neo4j AuraDB ---
NEO4J_URI = ""
NEO4J_USERNAME = ""
NEO4J_PASSWORD = ""

# Batched writes to Neo4j to avoid huge transactions
NEO4J_NODE_BATCH_SIZE = 500
NEO4J_REL_BATCH_SIZE = 1000

# If you want to create vector indexes as well
CREATE_VECTOR_INDEXES = True
# text-embedding-3-small default dimension is 1536
EMBEDDING_DIMENSIONS = 1536


# =========================
# HELPERS
# =========================

def chunked(iterable: Iterable, size: int):
    """Yield lists of up to `size` items from `iterable`."""
    iterable = list(iterable)
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(client: OpenAI, texts: List[str], model: str, batch_size: int) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI in batches.
    Returns vectors in the same order as texts.
    """
    all_embeddings: List[List[float]] = []
    for batch in chunked(texts, batch_size):
        resp = client.embeddings.create(
            model=model,
            input=batch,
        )
        for item in resp.data:
            all_embeddings.append([float(x) for x in item.embedding])
    return all_embeddings


# =========================
# DATA LOADING & PREP
# =========================

def load_and_prepare_data() -> Dict[str, object]:
    """Load CSV files and prepare unique nodes + relationships."""

    kh = pd.read_csv(KNOWLEDGE_HIERARCHY_CSV_PATH)

    # Normalize relevant text columns
    text_cols = [
        "ESCODiscipline",
        "ESCOBoK",
        "ESCOBoK Description",
        "ESCOKnowledgeConcept",
        "ESCOKnowledgeConcept Description",
    ]
    for col in text_cols:
        if col in kh.columns:
            kh[col] = kh[col].fillna("").astype(str).str.strip()

    # --- ESCODiscipline nodes ---
    discipline_labels = sorted(
        {l for l in kh["ESCODiscipline"].unique() if isinstance(l, str) and l.strip()}
    )

    # --- ESCOBoK nodes ---
    bok_df = kh[["ESCOBoK", "ESCOBoK Description"]].copy()
    bok_df = bok_df[bok_df["ESCOBoK"] != ""]
    bok_df = bok_df.drop_duplicates(subset=["ESCOBoK"]).reset_index(drop=True)

    bok_map = {
        row["ESCOBoK"]: row["ESCOBoK Description"] or ""
        for _, row in bok_df.iterrows()
    }

    # --- ESCOKnowledgeConcept nodes ---
    kc_df = kh[["ESCOKnowledgeConcept", "ESCOKnowledgeConcept Description"]].copy()
    kc_df = kc_df[kc_df["ESCOKnowledgeConcept"] != ""]
    kc_df = kc_df.drop_duplicates(subset=["ESCOKnowledgeConcept"]).reset_index(drop=True)

    kc_map = {
        row["ESCOKnowledgeConcept"]: row["ESCOKnowledgeConcept Description"] or ""
        for _, row in kc_df.iterrows()
    }

    # --- Relationships from knowledgeHierarchy ---
    discipline_to_bok = set()
    bok_to_kc = set()

    for _, row in kh.iterrows():
        d_label = row.get("ESCODiscipline", "").strip()
        b_label = row.get("ESCOBoK", "").strip()
        k_label = row.get("ESCOKnowledgeConcept", "").strip()

        if d_label and b_label:
            discipline_to_bok.add((d_label, b_label))
        if b_label and k_label:
            bok_to_kc.add((b_label, k_label))

    # --- 'consistsOfKnowledgeUnit' from mapping CSV ---
    kb_map = pd.read_csv(KNOWLEDGE_MAPPING_CSV_PATH)

    kb_map["preferredLabel"] = kb_map["preferredLabel"].fillna("").astype(str).str.strip()
    kb_map["broaderpreferredLabel"] = kb_map["broaderpreferredLabel"].fillna("").astype(str).str.strip()

    existing_kc_labels = set(kc_map.keys())

    consists_rels = set()
    for _, row in kb_map.iterrows():
        narrower = row["preferredLabel"]
        broader = row["broaderpreferredLabel"]
        if not narrower or not broader:
            continue
        if broader in existing_kc_labels and narrower in existing_kc_labels:
            # Edge from broader (column 2) to narrower (column 1)
            consists_rels.add((broader, narrower))

    return {
        "discipline_labels": discipline_labels,
        "bok_map": bok_map,
        "kc_map": kc_map,
        "discipline_to_bok": discipline_to_bok,
        "bok_to_kc": bok_to_kc,
        "consists_rels": consists_rels,
    }


# =========================
# NEO4J SCHEMA CREATION
# =========================

def create_constraints_and_indexes(driver):
    """Create uniqueness constraints and vector indexes."""
    constraint_cypher_statements = [
        # Unique labels
        """
        CREATE CONSTRAINT escodiscipline_unique IF NOT EXISTS
        FOR (d:ESCODiscipline)
        REQUIRE d.ESCODisLabel IS UNIQUE
        """,
        """
        CREATE CONSTRAINT escobok_unique IF NOT EXISTS
        FOR (b:ESCOBoK)
        REQUIRE b.ESCOBoKLabel IS UNIQUE
        """,
        """
        CREATE CONSTRAINT escoknowledgeconcept_unique IF NOT EXISTS
        FOR (k:ESCOKnowledgeConcept)
        REQUIRE k.ESCOKnowCon IS UNIQUE
        """,
    ]

    vector_index_cypher_statements = [
        f"""
        CREATE VECTOR INDEX escodiscipline_embedding_index IF NOT EXISTS
        FOR (d:ESCODiscipline)
        ON (d.ESCODisEmbdng)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIMENSIONS},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """,
        f"""
        CREATE VECTOR INDEX escobok_embedding_index IF NOT EXISTS
        FOR (b:ESCOBoK)
        ON (b.ESCOBoKEmbdng)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIMENSIONS},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """,
        f"""
        CREATE VECTOR INDEX escoknowledgeconcept_embedding_index IF NOT EXISTS
        FOR (k:ESCOKnowledgeConcept)
        ON (k.ESCOKnowConEmbdng)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIMENSIONS},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """,
        # NEW: index only for ESCOKnowledgeConcept nodes that also
        # have label ESCOKnowledgeGroup (i.e. nodes with outgoing consistsOfKnowledgeUnit)
        f"""
        CREATE VECTOR INDEX escoknowledgegroup_embedding_index IF NOT EXISTS
        FOR (k:ESCOKnowledgeGroup)
        ON (k.ESCOKnowConEmbdng)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {EMBEDDING_DIMENSIONS},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """,
    ]

    with driver.session() as session:
        for stmt in constraint_cypher_statements:
            session.run(stmt)

        if CREATE_VECTOR_INDEXES:
            for stmt in vector_index_cypher_statements:
                try:
                    session.run(stmt)
                except Exception as e:
                    print("Warning: could not create vector index.")
                    print("  Statement:", stmt)
                    print("  Error:", e)


# =========================
# NEO4J DATA INGESTION
# =========================

def upsert_escodiscipline_nodes(driver, nodes: List[Dict]):
    cypher = """
    UNWIND $rows AS row
    MERGE (d:ESCODiscipline {ESCODisLabel: row.label})
    SET d.ESCODisEmbdng = row.embedding
    """
    with driver.session() as session:
        for batch in chunked(nodes, NEO4J_NODE_BATCH_SIZE):
            session.run(cypher, rows=batch)


def upsert_escobok_nodes(driver, nodes: List[Dict]):
    cypher = """
    UNWIND $rows AS row
    MERGE (b:ESCOBoK {ESCOBoKLabel: row.label})
    SET b.ESCOBoKDes = row.description,
        b.ESCOBoKEmbdng = row.embedding
    """
    with driver.session() as session:
        for batch in chunked(nodes, NEO4J_NODE_BATCH_SIZE):
            session.run(cypher, rows=batch)


def upsert_escoknowledgeconcept_nodes(driver, nodes: List[Dict]):
    cypher = """
    UNWIND $rows AS row
    MERGE (k:ESCOKnowledgeConcept {ESCOKnowCon: row.label})
    SET k.ESCOKnowConDes = row.description,
        k.ESCOKnowConEmbdng = row.embedding
    """
    with driver.session() as session:
        for batch in chunked(nodes, NEO4J_NODE_BATCH_SIZE):
            session.run(cypher, rows=batch)


def create_hasBoK_relationships(driver, rels: List[Dict]):
    cypher = """
    UNWIND $rows AS row
    MATCH (d:ESCODiscipline {ESCODisLabel: row.discipline})
    MATCH (b:ESCOBoK {ESCOBoKLabel: row.bok})
    MERGE (d)-[:hasBoK]->(b)
    """
    with driver.session() as session:
        for batch in chunked(rels, NEO4J_REL_BATCH_SIZE):
            session.run(cypher, rows=batch)


def create_hasKnowledgeElement_relationships(driver, rels: List[Dict]):
    cypher = """
    UNWIND $rows AS row
    MATCH (b:ESCOBoK {ESCOBoKLabel: row.bok})
    MATCH (k:ESCOKnowledgeConcept {ESCOKnowCon: row.kc})
    MERGE (b)-[:hasKnowledgeElement]->(k)
    """
    with driver.session() as session:
        for batch in chunked(rels, NEO4J_REL_BATCH_SIZE):
            session.run(cypher, rows=batch)


def create_consistsOfKnowledgeUnit_relationships(driver, rels: List[Dict]):
    cypher = """
    UNWIND $rows AS row
    MATCH (broader:ESCOKnowledgeConcept {ESCOKnowCon: row.broader})
    MATCH (narrower:ESCOKnowledgeConcept {ESCOKnowCon: row.narrower})
    MERGE (broader)-[:consistsOfKnowledgeUnit]->(narrower)
    """
    with driver.session() as session:
        for batch in chunked(rels, NEO4J_REL_BATCH_SIZE):
            session.run(cypher, rows=batch)


def tag_knowledge_groups(driver):
    """
    Add the ESCOKnowledgeGroup label to any ESCOKnowledgeConcept
    node that has an outgoing :consistsOfKnowledgeUnit relationship.

    Before: (:ESCOKnowledgeConcept {...})
    After:  (:ESCOKnowledgeConcept:ESCOKnowledgeGroup {...})
    """
    cypher = """
    MATCH (k:ESCOKnowledgeConcept)-[:consistsOfKnowledgeUnit]->(:ESCOKnowledgeConcept)
    SET k:ESCOKnowledgeGroup
    """
    with driver.session() as session:
        session.run(cypher)


# =========================
# MAIN
# =========================

def main():
    # --- Load and prepare data ---
    data = load_and_prepare_data()

    discipline_labels: List[str] = data["discipline_labels"]
    bok_map: Dict[str, str] = data["bok_map"]
    kc_map: Dict[str, str] = data["kc_map"]
    discipline_to_bok = data["discipline_to_bok"]
    bok_to_kc = data["bok_to_kc"]
    consists_rels = data["consists_rels"]

    print(f"# Unique ESCODiscipline labels: {len(discipline_labels)}")
    print(f"# Unique ESCOBoK labels: {len(bok_map)}")
    print(f"# Unique ESCOKnowledgeConcept labels: {len(kc_map)}")
    print(f"# discipline -> BoK relationships: {len(discipline_to_bok)}")
    print(f"# BoK -> KnowledgeConcept relationships: {len(bok_to_kc)}")
    print(f"# consistsOfKnowledgeUnit relationships: {len(consists_rels)}")

    # --- OpenAI client & embeddings ---
    client = get_openai_client()

    # 1) ESCODiscipline embeddings (label only)
    print("Embedding ESCODiscipline labels...")
    discipline_embeddings = embed_texts(
        client,
        discipline_labels,
        model=EMBEDDING_MODEL,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    discipline_nodes = [
        {"label": label, "embedding": emb}
        for label, emb in zip(discipline_labels, discipline_embeddings)
    ]

    # 2) ESCOBoK embeddings (label + description)
    print("Embedding ESCOBoK (label + description)...")
    bok_labels = list(bok_map.keys())
    bok_texts = [
        f"{label}. {bok_map[label]}" if bok_map[label] else label
        for label in bok_labels
    ]
    bok_embeddings = embed_texts(
        client,
        bok_texts,
        model=EMBEDDING_MODEL,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    bok_nodes = [
        {
            "label": label,
            "description": bok_map[label],
            "embedding": emb,
        }
        for label, emb in zip(bok_labels, bok_embeddings)
    ]

    # 3) ESCOKnowledgeConcept embeddings (label + description)
    print("Embedding ESCOKnowledgeConcept (label + description)...")
    kc_labels = list(kc_map.keys())
    kc_texts = [
        f"{label}. {kc_map[label]}" if kc_map[label] else label
        for label in kc_labels
    ]
    kc_embeddings = embed_texts(
        client,
        kc_texts,
        model=EMBEDDING_MODEL,
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    kc_nodes = [
        {
            "label": label,
            "description": kc_map[label],
            "embedding": emb,
        }
        for label, emb in zip(kc_labels, kc_embeddings)
    ]

    # --- Neo4j ingestion ---
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
    )

    try:
        print("Creating constraints and (optionally) vector indexes...")
        create_constraints_and_indexes(driver)

        print("Upserting ESCODiscipline nodes...")
        upsert_escodiscipline_nodes(driver, discipline_nodes)

        print("Upserting ESCOBoK nodes...")
        upsert_escobok_nodes(driver, bok_nodes)

        print("Upserting ESCOKnowledgeConcept nodes...")
        upsert_escoknowledgeconcept_nodes(driver, kc_nodes)

        # Prepare relationships
        discipline_bok_rels = [
            {"discipline": d, "bok": b} for (d, b) in discipline_to_bok
        ]
        bok_kc_rels = [
            {"bok": b, "kc": k} for (b, k) in bok_to_kc
        ]
        consists_rels_rows = [
            {"broader": broad, "narrower": narrow}
            for (broad, narrow) in consists_rels
        ]

        print("Creating :hasBoK relationships...")
        create_hasBoK_relationships(driver, discipline_bok_rels)

        print("Creating :hasKnowledgeElement relationships...")
        create_hasKnowledgeElement_relationships(driver, bok_kc_rels)

        print("Creating :consistsOfKnowledgeUnit relationships...")
        create_consistsOfKnowledgeUnit_relationships(driver, consists_rels_rows)

        print("Tagging ESCOKnowledgeGroup nodes (concepts with outgoing consistsOfKnowledgeUnit)...")
        tag_knowledge_groups(driver)

        print("Done. Your ESCO knowledge graph (with group index) is now in AuraDB.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
