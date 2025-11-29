"""
Requirements:
    pip install PyPDF2 requests beautifulsoup4 neo4j openai

This script:
  - Downloads an RTU programme PDF from a URL
  - Parses programme metadata
  - Follows RTU course links found in the PDF
  - Creates nodes/edges in AuraDB:
        :RTUStudyField
        :RTUProgram
        :RTUCourse
        :RTUTopic
        :hasStudyField
        :hasCourse
        :hasTopic
  - Stores embeddings using OpenAI text-embedding-3-small in properties:
        RTUProgramEmbdng
        RTUCourseEmbdng
        RTUTopicEmbdng
        RTUStudyFieldEmbdng
"""

import re
from typing import Dict, List, Any, Optional
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from neo4j import GraphDatabase
from openai import OpenAI


# ---------- CONFIG: FILL THESE IN ---------- #

NEO4J_URI = ""
NEO4J_USER = ""
NEO4J_PASSWORD = ""

OPENAI_API_KEY = ""
EMBEDDING_MODEL = "text-embedding-3-small"

# URL of the programme PDF (example: Business Informatics in English)
PROGRAM_PDF_URL = "https://stud.rtu.lv/rtu/spr_export/prog_pdf_en.56"

# ------------------------------------------- #

openai_client = OpenAI(api_key=OPENAI_API_KEY)
_embedding_cache: Dict[str, List[float]] = {}


# ------------ OPENAI EMBEDDINGS ------------ #


def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get an embedding for `text` using text-embedding-3-small.
    Uses a simple in-memory cache to avoid duplicate calls.
    """
    text = (text or "").strip()
    if not text:
        return None

    if text in _embedding_cache:
        return _embedding_cache[text]

    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    )
    emb = resp.data[0].embedding
    _embedding_cache[text] = emb
    return emb


# ------------ PDF DOWNLOAD & PARSING ------------ #


def download_program_pdf(url: str) -> bytes:
    """Download the programme PDF from the given URL."""
    print(f"Downloading programme PDF from: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.content


def extract_text_and_links_from_pdf_bytes(pdf_bytes: bytes) -> tuple[str, List[str]]:
    """
    From PDF bytes:
      - Extract all text as one big normalized string
      - Extract all RTU course links (stud.rtu.lv/rtu/discpub/...)
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    pages_text: List[str] = []
    links = set()
    url_pattern = re.compile(r"https?://\S+")

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

        # Link annotations
        annots = page.get("/Annots", [])
        for annot in annots or []:
            try:
                obj = annot.get_object()
            except Exception:
                continue
            action = obj.get("/A")
            if not action:
                continue
            uri = action.get("/URI")
            if isinstance(uri, str):
                links.add(uri.strip())

        # Fallback: search in text for any URLs
        for match in url_pattern.findall(page_text):
            url = match.rstrip(").,]\">")  # strip common trailing punctuation
            links.add(url)

    full_text = "\n".join(pages_text)
    full_text = re.sub(r"\s+", " ", full_text)

    # Only keep RTU discpub course links
    course_links = [
        u for u in links
        if "stud.rtu.lv" in u and "/rtu/discpub/" in u
    ]

    return full_text, sorted(course_links)


# ------------ SECTION SLICING HELPERS ------------ #


def extract_between_labels(
    text: str,
    start_label: str,
    end_labels: List[str],
) -> str:
    """
    Extract text that appears after `start_label` up to the earliest of `end_labels`.
    Case-insensitive; returns '' if start_label is not found.
    """
    # find start
    start_pattern = re.compile(re.escape(start_label), re.IGNORECASE)
    start_match = start_pattern.search(text)
    if not start_match:
        return ""

    start_idx = start_match.end()

    # find closest end label
    end_idx = len(text)
    for end_label in end_labels:
        end_pattern = re.compile(re.escape(end_label), re.IGNORECASE)
        m = end_pattern.search(text, pos=start_idx)
        if m:
            end_idx = min(end_idx, m.start())

    value = text[start_idx:end_idx]
    value = value.strip(" :-\u2013\u2014")  # strips spaces and common dash characters
    return value.strip()


def extract_programme_data_from_text(text: str) -> Dict[str, str]:
    """
    Extract the fields for one programme from the PDF text:
        - Title
        - Identification code (e.g. 'DMB0')
        - Higher Education Study Field
        - Abstract
        - Aims
    """
    title = extract_between_labels(
        text,
        "Title",
        ["Identification code"],
    )

    # Identification code line -> take only first token (e.g. 'DMB0')
    raw_identification = extract_between_labels(
        text,
        "Identification code",
        [
            "Education classification code",
            "Higher education study field",
            "Duration of studies",
            "Study programme type",
            "Study programme director",
        ],
    )
    identification_code = raw_identification.split()[0] if raw_identification else ""

    higher_ed_field = extract_between_labels(
        text,
        "Higher education study field",
        ["Head of the study field", "Head of the study programme", "Department responsible"],
    )

    abstract = extract_between_labels(
        text,
        "Abstract",
        ["Aim"],
    )

    aims = extract_between_labels(
        text,
        "Aim",  # label in the PDF is singular "Aim"
        ["Tasks", "Learning outcomes", "Final/state examination procedure", "Final examination procedure"],
    )

    def squash_spaces(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    return {
        "Title": squash_spaces(title),
        "Identification code": squash_spaces(identification_code),
        "Higher Education Study Field": squash_spaces(higher_ed_field),
        "Abstract": squash_spaces(abstract),
        "Aims": squash_spaces(aims),
    }


# ------------ COURSE PAGE SCRAPING ------------ #


def parse_course_page(url: str) -> Dict[str, Any]:
    """
    Parse a single RTU course page.

    Returns dict:
      {
        "code": str,
        "name": str,
        "field_of_study": str,
        "annotation": str,
        "topics": List[str]   # one entry per content topic/row
      }
    """
    print(f"Fetching course page: {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Flatten all text into lines
    text = soup.get_text("\n", strip=True)
    text = text.replace("\xa0", " ")  # normalize non-breaking spaces

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # ---------- BASIC FIELDS (Code, Name, Field of study) ----------

    code = ""
    name = ""
    field_of_study = ""

    for i, l in enumerate(lines):
        # Value is on the NEXT line
        if l == "Code" and i + 1 < len(lines):
            code = lines[i + 1].strip()
        elif l == "Name" and i + 1 < len(lines):
            name = lines[i + 1].strip()
        elif l == "Field of study" and i + 1 < len(lines):
            field_of_study = lines[i + 1].strip()

    # ---------- ANNOTATION ----------

    annotation = ""
    for i, l in enumerate(lines):
        if l == "Annotation":
            annotation_lines: List[str] = []
            # everything after "Annotation" until the next section heading
            stop_markers = {
                "Contents",
                "Content",
                "Goals and objectives",
                "Learning outcomes",
                "Evaluation criteria of study results",
                "Course prerequisites",
                "Course planning",
            }
            for j in range(i + 1, len(lines)):
                nxt = lines[j]
                if nxt in stop_markers:
                    break
                annotation_lines.append(nxt)
            annotation = " ".join(annotation_lines).strip()
            break

    # ---------- CONTENTS (topics) ----------

    contents: List[str] = []

    header_lines = {
        "Content",
        "Full- and part-time intramural studies",
        "Part time extramural studies",
        "Contact hours",
        "Independent work",
    }

    content_start_idx = None
    for i, l in enumerate(lines):
        if l == "Contents":
            content_start_idx = i
            break

    if content_start_idx is not None:
        for i in range(content_start_idx + 1, len(lines)):
            l = lines[i]

            # End of the contents table
            if l.startswith("Total:") or l == "Total":
                break

            # Skip header labels
            if l in header_lines:
                continue

            # Skip pure numeric cells (hours, etc.)
            num_candidate = l.replace(".", "", 1)
            if num_candidate.isdigit():
                continue

            topic = l.strip()
            if topic:
                contents.append(topic)

    # Optional dedup within a single course
    seen = set()
    unique_topics = []
    for t in contents:
        if t not in seen:
            seen.add(t)
            unique_topics.append(t)

    return {
        "code": code,
        "name": name,
        "field_of_study": field_of_study,
        "annotation": annotation,
        "topics": unique_topics,
    }


# ------------ NEO4J WRITE HELPERS ------------ #


def create_program_and_study_field(
    tx,
    program_data: Dict[str, str],
    program_embedding: Optional[List[float]],
    study_field: str,
    study_field_embedding: Optional[List[float]],
):
    """
    Create/merge RTUProgram and RTUStudyField, plus :hasStudyField.
    """
    tx.run(
        """
        MERGE (p:RTUProgram {RTUProgramID: $program_id})
        SET p.RTUProgramTitle    = $title,
            p.RTUProgramAbstract = $abstract,
            p.RTUProgramAims     = $aims,
            p.RTUProgramEmbdng   = $embedding,
            p.RTUProgramMapped   = coalesce(p.RTUProgramMapped, 'no')

        WITH p, $study_field AS sf_label, $study_field_embedding AS sf_emb
        WHERE sf_label IS NOT NULL AND sf_label <> ''
        MERGE (sf:RTUStudyField {RTUStudyFieldLabel: sf_label})
        SET sf.RTUStudyFieldEmbdng = sf_emb,
            sf.RTUStudyFieldMapped = coalesce(sf.RTUStudyFieldMapped, 'no')
        MERGE (p)-[:hasStudyField]->(sf)
        """,
        program_id=program_data["Identification code"],
        title=program_data["Title"],
        abstract=program_data["Abstract"],
        aims=program_data["Aims"],
        embedding=program_embedding,
        study_field=study_field,
        study_field_embedding=study_field_embedding,
    )


def create_course_and_topics(
    tx,
    program_id: str,
    course: Dict[str, Any],
    course_embedding: Optional[List[float]],
    topics: List[Dict[str, Any]],
):
    """
    Create/merge RTUCourse and RTUTopic nodes, plus :hasCourse and :hasTopic edges.
    """
    tx.run(
        """
        MATCH (p:RTUProgram {RTUProgramID: $program_id})
        MERGE (c:RTUCourse {RTUCourseCode: $code})
        SET c.RTUCourseTitle        = $title,
            c.RTUCourseFieldofStudy = $field_of_study,
            c.RTUCourseAnnotation   = $annotation,
            c.RTUCourseEmbdng       = $course_embedding,
            c.RTUCourseMapped       = coalesce(c.RTUCourseMapped, 'no')
        MERGE (p)-[:hasCourse]->(c)

        WITH c, $topics AS topic_list
        UNWIND topic_list AS t
        MERGE (topic:RTUTopic {RTUTopicLabel: t.label})
        SET topic.RTUTopicEmbdng = t.embedding,
            topic.RTUTopicMapped = coalesce(topic.RTUTopicMapped, 'no')
        MERGE (c)-[:hasTopic]->(topic)
        """,
        program_id=program_id,
        code=course["code"],
        title=course["name"],
        field_of_study=course["field_of_study"],
        annotation=course["annotation"],
        course_embedding=course_embedding,
        topics=topics,
    )


# ------------ MAIN PIPELINE ------------ #


def process_program_pdf_url(program_pdf_url: str):
    # Download PDF
    pdf_bytes = download_program_pdf(program_pdf_url)

    # Extract text + course links
    pdf_text, course_urls = extract_text_and_links_from_pdf_bytes(pdf_bytes)

    # Extract programme data from text
    program_data = extract_programme_data_from_text(pdf_text)
    program_id = program_data.get("Identification code") or ""

    if not program_id:
        print("WARNING: No Identification code found in programme PDF; aborting.")
        return

    # Prepare embeddings
    program_text_for_emb = " ".join(
        x
        for x in [
            program_data.get("Title", ""),
            program_data.get("Abstract", ""),
            program_data.get("Aims", ""),
        ]
        if x
    )
    program_embedding = get_embedding(program_text_for_emb)

    study_field_label = program_data.get("Higher Education Study Field", "")
    study_field_embedding = get_embedding(study_field_label) if study_field_label else None

    # Connect to Neo4j
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    with driver:
        with driver.session() as session:
            # Create program + study field
            session.execute_write(
                create_program_and_study_field,
                program_data,
                program_embedding,
                study_field_label,
                study_field_embedding,
            )
            print(f"Upserted RTUProgram {program_id}")

            if course_urls:
                print(f"Found {len(course_urls)} course link(s) in programme PDF.")
            else:
                print("No course links found in programme PDF.")
                return

            seen_urls = set()

            for url in course_urls:
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                try:
                    course = parse_course_page(url)
                except Exception as e:
                    print(f"  Error fetching/parsing course at {url}: {e}")
                    continue

                if not course.get("code"):
                    print(f"  WARNING: Course at {url} has no code; skipping.")
                    continue

                # Course embedding: Title + Annotation
                course_text_for_emb = " ".join(
                    x
                    for x in [
                        course.get("name", ""),
                        course.get("annotation", ""),
                    ]
                    if x
                )
                course_embedding = get_embedding(course_text_for_emb)

                # Topic embeddings
                topics_param: List[Dict[str, Any]] = []
                for topic_label in course.get("topics", []):
                    topic_label_clean = topic_label.strip()
                    if not topic_label_clean:
                        continue
                    topic_embedding = get_embedding(topic_label_clean)
                    topics_param.append(
                        {
                            "label": topic_label_clean,
                            "embedding": topic_embedding,
                        }
                    )

                session.execute_write(
                    create_course_and_topics,
                    program_id,
                    course,
                    course_embedding,
                    topics_param,
                )
                print(f"  Upserted RTUCourse {course['code']} with {len(topics_param)} topic(s)")

    print("All done.")


if __name__ == "__main__":
    process_program_pdf_url(PROGRAM_PDF_URL)
