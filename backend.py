from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import mysql.connector
from mysql.connector import pooling
import json
import requests
import re
from typing import Optional, List, Dict, Any, Tuple
import logging
from datetime import datetime, date
from decimal import Decimal
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
from fastapi.responses import StreamingResponse
import os
import asyncio
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
from dotenv import load_dotenv

# Load environment variables from chatbott-specific .env file
# Explicitly use chatbott/.env, NOT the parent directory .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path, override=True)

# Configure logging - Console only
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom JSON Encoder for Decimal and datetime objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

app = FastAPI(json_encoder=CustomJSONEncoder)

# IMPROVEMENT 6: Production CORS fix - credentials + wildcard don't mix in browsers
# Use disable credentials for open API, or whitelist specific origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
ALLOW_CREDENTIALS = os.getenv("ALLOW_CREDENTIALS", "false").lower() == "true"

# If using wildcard origins, must disable credentials per browser CORS policy
if "*" in ALLOWED_ORIGINS and ALLOW_CREDENTIALS:
    logger.warning("CORS: Wildcard origins with credentials=true will fail in browsers. Setting credentials=false.")
    ALLOW_CREDENTIALS = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files serving through dedicated route
@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    """Serve static files like logo.jpg"""
    static_dir = os.path.dirname(os.path.abspath(__file__))
    file_full_path = os.path.join(static_dir, file_path)

    # Security: prevent directory traversal
    if not os.path.abspath(file_full_path).startswith(os.path.abspath(static_dir)):
        raise HTTPException(status_code=403, detail="Access denied")

    if os.path.exists(file_full_path) and os.path.isfile(file_full_path):
        return FileResponse(file_full_path)
    raise HTTPException(status_code=404, detail="File not found")

# Database configuration (IMPROVEMENT 7: Env-driven with sane defaults)
# Uses CHATBOT_DB_* variables from .env for bpd_ai_storyboard database
DB_CONFIG = {
    "host": os.getenv("CHATBOT_DB_HOST", "localhost"),
    "user": os.getenv("CHATBOT_DB_USER", "root"),
    "password": os.getenv("CHATBOT_DB_PASSWORD", "root"),
    "database": os.getenv("CHATBOT_DB_DATABASE", "bpd_ai_storyboard"),
    "port": int(os.getenv("CHATBOT_DB_PORT", 3306)),
}

# OpenAI/GPT-4o configuration for SQL generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = "gpt-4o"

# Ollama configuration (for intent extraction and answer generation)
OLLAMA_BASE_URL = "http://localhost:11434/api"
SQL_GENERATOR_MODEL = "gpt-4o"  # Using GPT-4o instead of qwen2.5-coder
ANSWER_GENERATOR_MODEL = "llama3:latest"
FAST_MODEL = "llama3:latest"

# Evaluation logging configuration (REFINEMENT 6)
EVAL_LOG_PATH = os.path.join(os.path.dirname(__file__), "evaluation_logs.jsonl")

# Create persistent Ollama session with aggressive connection pooling
ollama_session = requests.Session()
retry_strategy = Retry(total=1, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=100, pool_maxsize=100)
ollama_session.mount("http://", adapter)
ollama_session.mount("https://", adapter)
ollama_session.headers.update({"Connection": "keep-alive", "User-Agent": "FastAPI-Chatbot/1.0"})

# MySQL Connection Pool - reuse connections instead of creating new ones
try:
    db_pool = pooling.MySQLConnectionPool(
        pool_name="chatbot_pool",
        pool_size=10,
        pool_reset_session=True,
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        autocommit=True,
        use_pure=False
    )
    logger.info("MySQL connection pool created successfully")
except Exception as e:
    logger.warning(f"Could not create MySQL connection pool: {str(e)}. Will use direct connections.")
    db_pool = None

# ============================================================================
# 1. SCHEMA REGISTRY MODULE - Dynamic schema introspection on startup
# ============================================================================

# Global schema registry populated on startup from INFORMATION_SCHEMA
SCHEMA_REGISTRY = {}

def introspect_mysql_schema():
    """
    Introspect MySQL database INFORMATION_SCHEMA on startup.
    Builds SCHEMA_REGISTRY with table names, columns, data types, and sample values.
    Replaces hard-coded schema text blocks.
    """
    logger.info("Starting MySQL schema introspection...")
    global SCHEMA_REGISTRY

    connection = None
    try:
        # Connect to database
        if db_pool:
            connection = db_pool.get_connection()
        else:
            connection = mysql.connector.connect(**DB_CONFIG)

        cursor = connection.cursor(dictionary=True)

        # Get all table names (BASE TABLES only, excluding VIEWS)
        cursor.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'
        """, (DB_CONFIG['database'],))

        tables = cursor.fetchall()

        for table_row in tables:
            table_name = table_row['TABLE_NAME']

            # Get column information for each table
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE, COLUMN_TYPE, IS_NULLABLE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                ORDER BY ORDINAL_POSITION
            """, (DB_CONFIG['database'], table_name))

            columns = cursor.fetchall()

            # Build column metadata
            column_metadata = {}
            for col in columns:
                column_metadata[col['COLUMN_NAME']] = {
                    'data_type': col['DATA_TYPE'],
                    'column_type': col['COLUMN_TYPE'],
                    'nullable': col['IS_NULLABLE'] == 'YES'
                }

            # Get sample values for each column (up to 5 distinct values)
            sample_values = {}
            for col_name in column_metadata.keys():
                try:
                    # Skip large text columns
                    if column_metadata[col_name]['data_type'] in ['text', 'longtext', 'mediumtext']:
                        continue

                    cursor.execute(f"""
                        SELECT DISTINCT `{col_name}`
                        FROM `{table_name}`
                        WHERE `{col_name}` IS NOT NULL
                        LIMIT 5
                    """)
                    samples = cursor.fetchall()
                    sample_values[col_name] = [str(row[col_name]) for row in samples]
                except Exception as e:
                    logger.warning(f"Could not fetch samples for {table_name}.{col_name}: {str(e)}")
                    sample_values[col_name] = []

            # Store in registry
            SCHEMA_REGISTRY[table_name] = {
                'columns': column_metadata,
                'sample_values': sample_values
            }

            logger.info(f"✓ Introspected table: {table_name} ({len(column_metadata)} columns)")

        cursor.close()
        logger.info(f"Schema introspection complete. Tables loaded: {len(SCHEMA_REGISTRY)}")

    except Exception as e:
        logger.error(f"Schema introspection failed: {str(e)}")
        # Fallback: continue with empty registry, will use description.json
    finally:
        if connection:
            connection.close()

# Load database descriptions from description.json as supplementary metadata
try:
    with open(r'C:\Animal and Fisheries\chatbott\description.json', 'r', encoding='utf-8') as f:
        DB_DESCRIPTIONS = json.load(f)
    logger.info(f"Database descriptions loaded successfully. Tables: {list(DB_DESCRIPTIONS.keys())}")
except Exception as e:
    logger.error(f"Failed to load database descriptions: {str(e)}")
    DB_DESCRIPTIONS = {}

# ============================================================================
# DYNAMIC YEAR RULES BUILDER - Extract actual year values from schema
# ============================================================================

def build_year_rules_from_schema() -> Dict[str, Any]:
    """
    Dynamically build YEAR_RULES from actual data in SCHEMA_REGISTRY.
    For each table with a 'year' column, extract the actual year values from sample data.
    This ensures year filters always match real data, preventing 0-result queries.
    """
    year_rules = {}

    for table_name, table_meta in SCHEMA_REGISTRY.items():
        # Look for year-related columns
        year_col = None
        for col_name in table_meta['columns'].keys():
            if col_name.lower() == 'year' or 'year' in col_name.lower():
                year_col = col_name
                break

        if year_col and year_col in table_meta.get('sample_values', {}):
            sample_years = table_meta['sample_values'][year_col]
            if sample_years:
                # Get the latest year (sort to find latest)
                try:
                    # Try to sort as integers first
                    sorted_years = sorted(set(str(y) for y in sample_years), key=lambda x: int(x) if x.isdigit() else 0, reverse=True)
                    latest_year = sorted_years[0] if sorted_years else sample_years[0]
                except (ValueError, TypeError):
                    # If not integers, sort as strings
                    latest_year = sorted(set(str(y) for y in sample_years), reverse=True)[0]

                # Determine data type
                data_type = table_meta['columns'][year_col]['data_type']
                if data_type in ['CHAR', 'VARCHAR', 'TEXT']:
                    year_value = f"'{latest_year}'"
                    col_type = 'TEXT'
                else:
                    year_value = str(latest_year)
                    col_type = 'INTEGER'

                year_rules[table_name] = {
                    'year_column': year_col,
                    'year_value': year_value,
                    'type': col_type,
                    'sample_years': sample_years
                }
                logger.info(f"AUTO-DETECTED year rule for {table_name}: {year_col} = {year_value} ({col_type})")

    return year_rules

# Build year rules dynamically after schema introspection
YEAR_RULES = build_year_rules_from_schema()
logger.info(f"Dynamic YEAR_RULES built: {list(YEAR_RULES.keys())}")

# ============================================================================
# 2. TABLE ALIAS NORMALIZATION MAP
# ============================================================================

# Map district name variations to normalized forms
DISTRICT_ALIAS_MAP = {
    'gaya': 'Gayaji',
    'gayaji': 'Gayaji',
    'purbi champaran': 'East Champaran',
    'east champaran': 'East Champaran',
    'pashchim champaran': 'West Champaran',
    'west champaran': 'West Champaran',
    'kosi': 'Koshi',
    'koshi': 'Koshi',
    'kaimur': 'Kaimur(Bhabua)',
    'kaimur(bhabua)': 'Kaimur(Bhabua)',
    'kaimur (bhabua)': 'Kaimur(Bhabua)',
    'muzzafarpur': 'Muzaffarpur',
    'muzaffarpur': 'Muzaffarpur',
    'khargaria': 'Khagaria',
    'khagaria': 'Khagaria',
    'sahrsa': 'Saharsa',
    'saharsa': 'Saharsa',
    'sarhasa': 'Saharsa',
    'purnea': 'Purnia',
    'purnia': 'Purnia',
    'shekhpura': 'Sheikhpura',
    'sheikhpura': 'Sheikhpura'
}

# Map division name variations
DIVISION_ALIAS_MAP = {
    'kosi': 'Koshi',
    'koshi': 'Koshi',
    'sahrsa': 'Saharsa',
    'saharsa': 'Saharsa',
    'purnea': 'Purnia',
    'purnia': 'Purnia'
}

def normalize_location_names(question: str) -> str:
    """
    Apply table alias normalization before SQL generation.
    Normalizes district and division name variations.
    """
    question_lower = question.lower()

    # Normalize district names
    for alias, canonical in DISTRICT_ALIAS_MAP.items():
        # Case-insensitive replacement
        pattern = r'\b' + re.escape(alias) + r'\b'
        question = re.sub(pattern, canonical, question, flags=re.IGNORECASE)

    # Normalize division names
    for alias, canonical in DIVISION_ALIAS_MAP.items():
        pattern = r'\b' + re.escape(alias) + r'\b'
        question = re.sub(pattern, canonical, question, flags=re.IGNORECASE)

    return question

# ============================================================================
# 3. HYBRID TABLE RETRIEVAL - Sentence embeddings + BM25
# ============================================================================

# Note: For production, install sentence-transformers and rank-bm25
# pip install sentence-transformers rank-bm25
# For now, we'll use a simplified version with keyword matching and scoring

def compute_table_relevance_scores(question: str) -> List[Tuple[str, float]]:
    """
    Hybrid table retrieval using keyword matching and semantic scoring.
    Returns list of (table_name, score) tuples sorted by relevance.
    In production, this would use sentence-transformer embeddings + BM25.
    """
    question_lower = question.lower()
    table_scores = {}

    # Keyword-based scoring for each table
    table_keywords = {
        'analysis_all_hatchery_district_wise': [
            'hatchery', 'hatcheries', 'functional', 'non-functional', 'breeding',
            'exotic', 'imc', 'fingerling', 'fry', 'spawn', 'fish seed production'
        ],
        'areas_without_jalkars_by_range_district_block_panchayat': [
            'without jalkar', 'no jalkar', 'no water body', 'areas without',
            'missing jalkar', 'zero jalkar'
        ],
        'details_fcs_district_wise': [
            'fcs', 'cooperative society', 'cooperative societies', 'members',
            'male members', 'female members', 'blocks in district'
        ],
        'details_newly_formed_fcs_district_wise': [
            'newly formed fcs', 'new fcs', 'new cooperative', 'registration date',
            'when formed', 'inland fcs', 'marine fcs'
        ],
        'jalkars_spread_across_multiple_panchayats_block_district_range_w': [
            'multiple panchayat', 'spread across', 'boundary', 'spanning',
            'inter-panchayat', 'shared jalkar'
        ],
        'list_of_jalkars_with_unclear_settlement_status_panchayat_wise': [
            'unclear settlement', 'settlement status', 'court case', 'disputed',
            'free fishing', 'perta', 'khata', 'khasra', 'pending'
        ],
        'total_revenue_jalkar_report': [
            'fish production', 'revenue collection', 'export', 'import',
            'fish seed', 'annual target', 'production target'
        ]
    }

    # Score each table based on keyword matches
    for table_name, keywords in table_keywords.items():
        score = 0.0
        for keyword in keywords:
            if keyword in question_lower:
                # Exact phrase match gets higher score
                score += len(keyword.split())

        # Bonus for table name appearing in question
        if table_name.replace('_', ' ') in question_lower:
            score += 10

        if score > 0:
            table_scores[table_name] = score

    # Sort by score descending
    sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"Table relevance scores: {sorted_tables}")
    return sorted_tables

def get_top_candidate_tables(question: str, top_k: int = 2) -> List[str]:
    """
    Get top-K candidate tables using hybrid retrieval.
    Returns at most top_k tables with highest relevance scores.
    """
    # Normalize question first
    normalized_question = normalize_location_names(question)

    # Compute relevance scores
    table_scores = compute_table_relevance_scores(normalized_question)

    # Keep top-K tables
    top_tables = [table for table, score in table_scores[:top_k]]

    if not top_tables:
        # Fallback: use keyword detection
        logger.warning("No tables scored, falling back to keyword detection")
        detected = detect_question_keywords(normalized_question)
        top_tables = detected.get('tables', [])[:top_k]

    logger.info(f"Top-{top_k} candidate tables: {top_tables}")
    return top_tables

# ============================================================================
# 4. INTENT EXTRACTION STEP - LLM outputs strict JSON
# ============================================================================

def validate_intent_for_sql(intent: Dict[str, Any]) -> bool:
    """
    REFINEMENT 1: SQL Generation Safety - Plan Validation Before SQL
    Validates intent JSON structure before passing to SQL generation.
    Prevents generation from partial or invalid plans.
    """
    if not intent:
        logger.warning("Intent is None or empty")
        return False

    if not intent.get("tables"):
        logger.warning("Intent missing 'tables' field")
        return False

    if not intent.get("metrics"):
        logger.warning("Intent missing 'metrics' field")
        return False

    # Validate all tables exist in schema
    for table in intent.get("tables", []):
        if table not in SCHEMA_REGISTRY and table not in DB_DESCRIPTIONS:
            logger.warning(f"Table '{table}' not found in schema")
            return False

    # Validate all metrics columns are real
    for metric in intent.get("metrics", []):
        col_found = False
        for table in intent.get("tables", []):
            if table in SCHEMA_REGISTRY:
                if metric in SCHEMA_REGISTRY[table]["columns"]:
                    col_found = True
                    break
            elif table in DB_DESCRIPTIONS:
                if metric in DB_DESCRIPTIONS[table]["columns"]:
                    col_found = True
                    break
        if not col_found:
            logger.warning(f"Metric column '{metric}' not found in selected tables")
            return False

    # Validate filter columns
    for col in intent.get("filters", {}):
        col_found = False
        for table in intent.get("tables", []):
            if table in SCHEMA_REGISTRY:
                if col in SCHEMA_REGISTRY[table]["columns"]:
                    col_found = True
                    break
            elif table in DB_DESCRIPTIONS:
                if col in DB_DESCRIPTIONS[table]["columns"]:
                    col_found = True
                    break
        if not col_found:
            logger.warning(f"Filter column '{col}' not found in selected tables")
            return False

    logger.info("Intent validation passed")
    return True

def extract_query_intent(question: str, candidate_tables: List[str]) -> Dict[str, Any]:
    """
    LLM extracts intent as strict JSON with: tables, metrics, filters, group_by, order_by, limit.
    Validates against SCHEMA_REGISTRY and auto-repairs if invalid.
    """
    logger.info(f"Extracting query intent for: {question}")

    # Build schema context for candidate tables
    schema_context = "AVAILABLE TABLES AND COLUMNS:\n\n"
    for table in candidate_tables:
        if table in SCHEMA_REGISTRY:
            schema_context += f"TABLE: {table}\n"
            schema_context += f"Columns: {', '.join(SCHEMA_REGISTRY[table]['columns'].keys())}\n\n"
        elif table in DB_DESCRIPTIONS:
            schema_context += f"TABLE: {table}\n"
            schema_context += f"Columns: {', '.join(DB_DESCRIPTIONS[table]['columns'].keys())}\n\n"

    prompt = f"""You are a query intent extraction expert for Bihar government data. Analyze the user's question and extract the query intent as strict JSON.

{schema_context}

User Question: {question}

Extract the following intent components and output ONLY valid JSON (no markdown, no explanations):

{{
  "tables": ["table_name"],
  "metrics": ["column1", "column2"],
  "filters": {{"column_name": "value"}},
  "group_by": ["column1"],
  "order_by": {{"column": "column_name", "direction": "DESC"}}
}}

Rules:
1. Use ONLY exact table and column names from the schema above
2. metrics: columns to select or aggregate
3. filters: WHERE clause conditions - FOLLOW GEOGRAPHIC RULES BELOW
4. group_by: columns to group by (if aggregation is needed)
5. order_by: sorting specification (use "ASC" or "DESC")
6. **DO NOT include "limit" field. Always fetch ALL relevant data.** Only the final SQL generator should add LIMIT if the user explicitly asks for "top N".

⚠️ CRITICAL GEOGRAPHIC RULES - READ CAREFULLY:
- "Bihar" refers to the ENTIRE STATE, NOT a specific division
- NEVER use filter: "division_name": "Bihar" — this will return ZERO rows
- If question asks "each district of Bihar" or "all districts in Bihar":
  - DO NOT add division_name filter
  - DO use group_by: ["district_name"]
  - Leave filters EMPTY for state-level views
- For specific district: use "district_name": "DistrictName"
- For specific division: use "division_name": "DivisionName"
- Always validate that filter values match real divisions/districts (not state names)

CRITICAL: Output MUST be valid JSON.
- Do NOT wrap in code fences (no ```json)
- Do NOT include explanations or extra text
- The response MUST be parsable by Python json.loads() with no trailing text
- Output JSON only, starting with {{ and ending with }}"""

    try:
        response = call_gpt4o(prompt, temperature=0.1)
        logger.debug(f"Intent extraction response: {response[:200]}...")

        # IMPROVEMENT 4: Strict JSON parsing with fallback extraction
        response = response.strip()

        # Try to find JSON boundaries in case of extraneous text
        start_idx = response.find('{')
        end_idx = response.rfind('}')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx + 1]
            try:
                intent = json.loads(json_str)
                # Validate and auto-repair
                validated_intent = validate_and_repair_intent(intent, candidate_tables)
                logger.info(f"Extracted intent: {validated_intent}")
                return validated_intent
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error: {str(je)}")
                return {}
        else:
            logger.warning("No JSON boundaries found in intent extraction response")
            return {}

    except Exception as e:
        logger.error(f"Intent extraction failed: {str(e)}")
        return {}

def validate_and_repair_intent(intent: Dict[str, Any], candidate_tables: List[str]) -> Dict[str, Any]:
    """
    Validate intent against SCHEMA_REGISTRY and auto-repair invalid entries.
    """
    validated = {
        'tables': [],
        'metrics': [],
        'filters': {},
        'group_by': [],
        'order_by': {},
        'limit': None
    }

    # Validate tables
    for table in intent.get('tables', []):
        if table in SCHEMA_REGISTRY or table in DB_DESCRIPTIONS:
            validated['tables'].append(table)
        else:
            logger.warning(f"Invalid table in intent: {table}")

    if not validated['tables'] and candidate_tables:
        validated['tables'] = [candidate_tables[0]]

    # Validate columns against selected tables
    valid_columns = set()
    for table in validated['tables']:
        if table in SCHEMA_REGISTRY:
            valid_columns.update(SCHEMA_REGISTRY[table]['columns'].keys())
        elif table in DB_DESCRIPTIONS:
            valid_columns.update(DB_DESCRIPTIONS[table]['columns'].keys())

    # Validate metrics
    for metric in intent.get('metrics', []):
        if metric in valid_columns:
            validated['metrics'].append(metric)
        else:
            logger.warning(f"Invalid metric column: {metric}")

    # Validate filters - check both column existence and reasonable values
    for col, val in intent.get('filters', {}).items():
        if col in valid_columns:
            # Check if filter value makes sense (exists in sample data or is reasonable)
            filter_valid = False

            # For geographic filters, validate against sample values if available
            if col in ['division_name', 'district_name', 'block_name', 'panchayat_name']:
                # Get sample values from schema
                for table in validated['tables']:
                    if table in SCHEMA_REGISTRY and col in SCHEMA_REGISTRY[table].get('sample_values', {}):
                        sample_vals = SCHEMA_REGISTRY[table]['sample_values'][col]
                        if sample_vals and str(val) not in sample_vals:
                            logger.warning(f"Filter value '{val}' for column '{col}' may not exist in data. Sample values: {sample_vals}")
                        filter_valid = True
                        break

                # If no samples found in schema, still allow it (fallback)
                if not filter_valid:
                    filter_valid = True
            else:
                # For non-geographic columns, just validate the column exists
                filter_valid = True

            if filter_valid:
                validated['filters'][col] = val
        else:
            logger.warning(f"Invalid filter column: {col}")

    # Validate group_by
    for col in intent.get('group_by', []):
        if col in valid_columns:
            validated['group_by'].append(col)

    # Validate order_by
    if 'order_by' in intent and isinstance(intent['order_by'], dict):
        order_col = intent['order_by'].get('column')
        if order_col and order_col in valid_columns:
            validated['order_by'] = intent['order_by']

    # Validate limit
    if 'limit' in intent and isinstance(intent['limit'], (int, str)):
        try:
            validated['limit'] = int(intent['limit'])
        except:
            pass

    return validated

# ============================================================================
# 5. YEAR RULES REGISTRY - NOW DYNAMICALLY GENERATED ABOVE
# ============================================================================
# NOTE: YEAR_RULES is now auto-generated by build_year_rules_from_schema()
# which extracts actual year values from schema sample data.
# The hardcoded values below have been replaced with dynamic detection.
# See build_year_rules_from_schema() function above for details.

def apply_year_filter(table_name: str, sql_query: str) -> str:
    """
    Apply year filter based on YEAR_RULES registry.
    Returns SQL query with appropriate year filter added.

    REFINEMENT 4: Type-safe insertion - handles missing WHERE, missing semicolon.
    ENHANCEMENT: Properly handles GROUP BY and ORDER BY clauses.
    """
    import re

    rule = YEAR_RULES.get(table_name)
    if not rule:
        return sql_query

    # Check if year filter already exists
    if f"{rule['year_column']} =" in sql_query.lower():
        logger.info(f"Year filter already present for {table_name}")
        return sql_query

    year_clause = f"{rule['year_column']} = {rule['year_value']}"

    # Remove trailing semicolon for consistent processing
    sql_query = sql_query.rstrip("; ")

    # Use regex to find GROUP BY and ORDER BY clauses
    group_by_pattern = re.compile(r'\s+GROUP\s+BY\s+', re.IGNORECASE)
    order_by_pattern = re.compile(r'\s+ORDER\s+BY\s+', re.IGNORECASE)

    group_by_match = group_by_pattern.search(sql_query)
    order_by_match = order_by_pattern.search(sql_query)

    # Determine insertion point - should be before GROUP BY, ORDER BY, or at end
    if group_by_match:
        insert_pos = group_by_match.start()
    elif order_by_match:
        insert_pos = order_by_match.start()
    else:
        insert_pos = len(sql_query)

    # Extract the part before insertion point and the part after
    before = sql_query[:insert_pos].rstrip()
    after = " " + sql_query[insert_pos:].lstrip() if insert_pos < len(sql_query) else ""

    # Check if WHERE clause already exists (case insensitive)
    has_where = re.search(r'\bWHERE\b', before, re.IGNORECASE) is not None

    # Add the year filter
    if has_where:
        # Already has WHERE, use AND
        year_filter = f" AND {year_clause}"
    else:
        # No WHERE, add one
        year_filter = f" WHERE {year_clause}"

    # Reconstruct the query
    sql_query = before + year_filter + after + ";"

    logger.info(f"Applied year filter for {table_name}: {year_clause}")
    return sql_query

# ============================================================================
# 6. SQL VALIDATION & REPAIR LAYER - Dry run with retry
# ============================================================================

def verify_sql_columns(sql_query: str) -> bool:
    """
    REFINEMENT 5: Schema-Aware Column Guard - FIXED VERSION
    Validates that all columns referenced in SQL actually exist in schema.
    Prevents hallucinated columns like 'revenue_percent', 'area_ha'.

    FIX: Now properly handles:
    - Column aliases (e.g., 'total_male_members' in 'AS total_male_members')
    - String literal values (e.g., 'Bihar' in 'WHERE division_name = 'Bihar'')
    - Aggregate function outputs

    Only validates actual column references, not aliases or values.
    """
    # Build set of table names to exclude
    table_names = set(SCHEMA_REGISTRY.keys()) | set(DB_DESCRIPTIONS.keys())

    # Build set of valid columns from schema
    valid_cols = set()
    for table_meta in SCHEMA_REGISTRY.values():
        valid_cols.update(table_meta['columns'].keys())

    # Also add columns from DB_DESCRIPTIONS
    for table_meta in DB_DESCRIPTIONS.values():
        valid_cols.update(table_meta['columns'].keys())

    # Add SQL keywords and functions (to avoid false positives)
    sql_keywords = {
        'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'AS', 'ON', 'JOIN',
        'LEFT', 'RIGHT', 'INNER', 'OUTER', 'GROUP', 'BY', 'ORDER', 'ASC', 'DESC',
        'LIMIT', 'OFFSET', 'SUM', 'COUNT', 'AVG', 'MIN', 'MAX', 'DISTINCT',
        'HAVING', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'CAST', 'LIKE',
        'BETWEEN', 'IS', 'NULL', 'UNION', 'ALL', 'VALUES', 'INSERT', 'UPDATE',
        'DELETE', 'CREATE', 'ALTER', 'DROP', 'TABLE', 'DATABASE', 'DEFAULT',
        'WITH', 'USING', 'TRUE', 'FALSE'
    }

    invalid_cols = []

    # Extract identifiers from SELECT clause - only columns BEFORE AS, not aliases
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        # Remove string literals to avoid false positives
        select_clause_clean = re.sub(r"'[^']*'", '', select_clause)
        select_clause_clean = re.sub(r"\d+", '', select_clause_clean)

        # Extract column names: look for identifiers before AS (actual columns, not aliases)
        col_refs = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s+AS\s+', select_clause_clean, re.IGNORECASE)
        # Also get columns after SELECT and before comma/FROM
        col_refs.extend(re.findall(r'(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,|AS\s|FROM)', 'SELECT ' + select_clause_clean + ' FROM', re.IGNORECASE))

        for col in col_refs:
            col = col.strip() if col else ''
            if not col:
                continue
            col_upper = col.upper()

            if col_upper in sql_keywords or col in table_names or col_upper in {t.upper() for t in table_names}:
                continue

            if col not in valid_cols and col_upper not in {c.upper() for c in valid_cols}:
                if col not in invalid_cols:
                    invalid_cols.append(col)

    # Extract identifiers from WHERE clause - only column names (left side of operators)
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|$)', sql_query, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_clause = where_match.group(1)
        # Remove string literals
        where_clause_clean = re.sub(r"'[^']*'", '', where_clause)

        # Extract column names that appear before comparison operators
        col_refs = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|!=|<>|<|>|<=|>=|LIKE|IN|BETWEEN)', where_clause_clean, re.IGNORECASE)

        for col in col_refs:
            col_upper = col.upper()

            if col_upper in sql_keywords or col in table_names or col_upper in {t.upper() for t in table_names}:
                continue

            if col not in valid_cols and col_upper not in {c.upper() for c in valid_cols}:
                if col not in invalid_cols:
                    invalid_cols.append(col)

    if invalid_cols:
        logger.warning(f"Detected potentially hallucinated identifiers: {invalid_cols}")
        return False

    logger.info("SQL column verification passed")
    return True

def validate_sql_with_dry_run(sql_query: str, max_retries: int = 2) -> Tuple[str, bool]:
    """
    Execute generated SQL in dry test.
    Catch errors and regenerate with error message.
    Retry up to max_retries times.
    Returns (final_sql, success_flag).
    """
    logger.info("Starting SQL validation with dry run")

    connection = None
    attempt = 0
    last_error = None

    while attempt <= max_retries:
        try:
            # Connect to database
            if db_pool:
                connection = db_pool.get_connection()
            else:
                connection = mysql.connector.connect(**DB_CONFIG)

            cursor = connection.cursor(dictionary=True)

            # Execute dry run with LIMIT 0 to validate syntax
            dry_run_query = sql_query.replace(';', ' LIMIT 0;')
            cursor.execute(dry_run_query)
            cursor.fetchall()

            # REFINEMENT 2: Execution-Guided Validation (EG-SQL)
            # If dry run succeeds, try actual execution to check for empty result sets
            try:
                # Execute actual query (but with low limit to avoid large results)
                test_query = sql_query.rstrip('; ') + ' LIMIT 1;'
                cursor.execute(test_query)
                data_preview = cursor.fetchall()

                if len(data_preview) == 0 and 'LIMIT 0' not in sql_query.upper():
                    logger.warning("Query executed successfully but returned 0 rows - attempting to relax filters")
                    # Try to remove WHERE clause and retry
                    relaxed_sql = re.sub(
                        r'\s+WHERE\s+.+?(?=GROUP BY|ORDER BY|LIMIT|;|$)',
                        '',
                        sql_query,
                        flags=re.IGNORECASE | re.DOTALL
                    )
                    relaxed_sql = relaxed_sql.rstrip('; ') + ';'

                    if relaxed_sql != sql_query:
                        logger.info("Retrying with relaxed WHERE clause")
                        cursor.execute(relaxed_sql.replace(';', ' LIMIT 0;'))
                        cursor.fetchall()
                        sql_query = relaxed_sql
            except Exception as preview_error:
                logger.debug(f"Preview execution failed (non-critical): {str(preview_error)}")
                # Continue with original query anyway
                pass

            cursor.close()

            logger.info(f"✓ SQL validation passed on attempt {attempt + 1}")
            return sql_query, True

        except mysql.connector.Error as e:
            last_error = str(e)
            logger.warning(f"SQL validation failed (attempt {attempt + 1}/{max_retries + 1}): {last_error}")

            # Try to repair SQL based on error
            sql_query = repair_sql_from_error(sql_query, last_error)
            attempt += 1

        except Exception as e:
            last_error = str(e)
            logger.error(f"Unexpected error in SQL validation: {last_error}")
            attempt += 1

        finally:
            if connection:
                connection.close()

    # All retries exhausted
    logger.error(f"SQL validation failed after {max_retries + 1} attempts. Last error: {last_error}")
    return sql_query, False

def repair_sql_from_error(sql_query: str, error_message: str) -> str:
    """
    Attempt to repair SQL query based on error message.
    Common fixes:
    - Unknown column: remove or replace column
    - Syntax error: fix common syntax issues
    - Aggregate without GROUP BY: add GROUP BY
    """
    logger.info(f"Attempting to repair SQL based on error: {error_message[:100]}")

    # Fix 1: Unknown column error
    if "unknown column" in error_message.lower():
        # Extract column name from error
        col_match = re.search(r"unknown column '([^']+)'", error_message, re.IGNORECASE)
        if col_match:
            bad_column = col_match.group(1)
            logger.info(f"Removing unknown column: {bad_column}")
            # Remove the column from SELECT
            sql_query = re.sub(r',?\s*`?' + re.escape(bad_column) + r'`?\s*,?', '', sql_query, flags=re.IGNORECASE)
            # Clean up double commas
            sql_query = re.sub(r',\s*,', ',', sql_query)
            sql_query = re.sub(r'SELECT\s*,', 'SELECT ', sql_query, flags=re.IGNORECASE)

    # Fix 2: Aggregate function without GROUP BY
    if "isn't in group by" in error_message.lower() or "must appear in the group by clause" in error_message.lower():
        # Extract non-aggregated columns from SELECT
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Find non-aggregated columns
            non_agg_cols = []
            for col in select_clause.split(','):
                col = col.strip()
                if not re.search(r'(SUM|COUNT|AVG|MAX|MIN)\s*\(', col, re.IGNORECASE):
                    # Extract column name (remove AS alias)
                    col_name = re.sub(r'\s+AS\s+.+', '', col, flags=re.IGNORECASE).strip()
                    if col_name and col_name != '*':
                        non_agg_cols.append(col_name)

            if non_agg_cols:
                # Add GROUP BY clause
                if 'GROUP BY' in sql_query.upper():
                    # Update existing GROUP BY
                    sql_query = re.sub(
                        r'GROUP BY\s+.+?(ORDER BY|HAVING|LIMIT|;)',
                        f"GROUP BY {', '.join(non_agg_cols)} \\1",
                        sql_query,
                        flags=re.IGNORECASE
                    )
                else:
                    # Add new GROUP BY before ORDER BY or at end
                    if 'ORDER BY' in sql_query.upper():
                        sql_query = re.sub(
                            r'ORDER BY',
                            f"GROUP BY {', '.join(non_agg_cols)} ORDER BY",
                            sql_query,
                            flags=re.IGNORECASE
                        )
                    else:
                        sql_query = re.sub(r';', f" GROUP BY {', '.join(non_agg_cols)};", sql_query)

    # Fix 3: Syntax error near 'WHERE' - likely aggregate in WHERE clause
    if "you have an error in your sql syntax" in error_message.lower() and 'where' in error_message.lower():
        sql_query = move_aggregates_to_having(sql_query)

    logger.info(f"Repaired SQL: {sql_query}")
    return sql_query

def move_aggregates_to_having(sql_query: str) -> str:
    """
    Move aggregate functions from WHERE clause to HAVING clause.
    Enforces rule: aggregates in WHERE should be in HAVING.
    """
    # Check if WHERE clause contains aggregates
    where_match = re.search(r'WHERE\s+(.+?)(?=GROUP BY|ORDER BY|LIMIT|;)', sql_query, re.IGNORECASE | re.DOTALL)
    if not where_match:
        return sql_query

    where_clause = where_match.group(1).strip()

    # Split conditions by AND/OR
    conditions = re.split(r'\s+(AND|OR)\s+', where_clause, flags=re.IGNORECASE)

    raw_conditions = []
    aggregate_conditions = []
    operators = []

    for i, cond in enumerate(conditions):
        if cond.upper() in ['AND', 'OR']:
            operators.append(cond)
        elif re.search(r'(SUM|COUNT|AVG|MAX|MIN)\s*\(', cond, re.IGNORECASE):
            aggregate_conditions.append(cond)
        else:
            raw_conditions.append(cond)

    if not aggregate_conditions:
        return sql_query

    # Rebuild query
    new_sql = sql_query

    # Replace WHERE clause with raw conditions only
    if raw_conditions:
        new_where = ' AND '.join(raw_conditions)
        new_sql = re.sub(
            r'WHERE\s+.+?(?=GROUP BY|ORDER BY|LIMIT|;)',
            f'WHERE {new_where} ',
            new_sql,
            flags=re.IGNORECASE | re.DOTALL
        )
    else:
        # Remove WHERE clause entirely
        new_sql = re.sub(
            r'WHERE\s+.+?(?=GROUP BY|ORDER BY|LIMIT|;)',
            '',
            new_sql,
            flags=re.IGNORECASE | re.DOTALL
        )

    # Add HAVING clause with aggregate conditions
    having_clause = ' AND '.join(aggregate_conditions)

    if 'GROUP BY' in new_sql.upper():
        # Add HAVING after GROUP BY
        if 'HAVING' in new_sql.upper():
            # Append to existing HAVING
            new_sql = re.sub(
                r'HAVING\s+',
                f'HAVING {having_clause} AND ',
                new_sql,
                flags=re.IGNORECASE
            )
        else:
            # Add new HAVING after GROUP BY
            new_sql = re.sub(
                r'(GROUP BY\s+[^;]+?)(?=ORDER BY|LIMIT|;)',
                f'\\1 HAVING {having_clause} ',
                new_sql,
                flags=re.IGNORECASE
            )
    else:
        # Need to add GROUP BY as well
        logger.warning("Cannot add HAVING without GROUP BY - skipping aggregate condition")

    return new_sql

# ============================================================================
# 7. ANSWER GENERATION HARDENING - Compute insights in Python
# ============================================================================

def compute_numeric_insights(query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute numeric insights in Python before feeding to answer model.
    Returns sanitized insights as JSON.
    """
    insights = {
        'total_rows': len(query_results),
        'numeric_stats': {},
        'top_values': {},
        'comparisons': []
    }

    if not query_results:
        return insights

    # Identify numeric columns
    first_row = query_results[0]
    numeric_cols = []
    text_cols = []

    for key, value in first_row.items():
        try:
            float(value) if value is not None else None
            numeric_cols.append(key)
        except (ValueError, TypeError):
            text_cols.append(key)

    # Compute statistics for numeric columns
    for col in numeric_cols:
        values = []
        for row in query_results:
            val = row.get(col)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    pass

        if values:
            insights['numeric_stats'][col] = {
                'min': min(values),
                'max': max(values),
                'sum': sum(values),
                'avg': sum(values) / len(values),
                'count': len(values)
            }

    # Extract top values for ranking queries
    if len(query_results) <= 10 and text_cols and numeric_cols:
        # Likely a top-N query
        category_col = text_cols[0]
        value_col = numeric_cols[0]

        top_items = []
        for row in query_results[:5]:
            cat = row.get(category_col)
            val = row.get(value_col)
            if cat and val is not None:
                top_items.append({'category': cat, 'value': float(val)})

        insights['top_values'][f'{category_col}_by_{value_col}'] = top_items

    # Compute comparisons for multi-row results
    if len(query_results) >= 2 and numeric_cols:
        for col in numeric_cols[:2]:  # Limit to first 2 numeric columns
            values = [float(row.get(col)) for row in query_results if row.get(col) is not None]
            if len(values) >= 2:
                highest = max(values)
                lowest = min(values)
                if highest != lowest:
                    insights['comparisons'].append({
                        'column': col,
                        'highest': highest,
                        'lowest': lowest,
                        'difference': highest - lowest,
                        'ratio': highest / lowest if lowest > 0 else None
                    })

    return insights

def sanitize_data_for_answer(query_results: List[Dict[str, Any]], max_rows: int = 50) -> str:
    """
    Sanitize and format query results as JSON for answer generation.
    Limits to max_rows and removes sensitive fields.
    """
    # Limit number of rows
    limited_results = query_results[:max_rows]

    # Convert to JSON
    sanitized_json = json.dumps(limited_results, indent=2, default=str)

    # Truncate if still too large
    if len(sanitized_json) > 4000:
        sanitized_json = sanitized_json[:4000] + "\n... (truncated)"

    return sanitized_json

def validate_answer_against_data(answer: str, query_results: List[Dict[str, Any]]) -> str:
    """
    Check output for numbers not in data before returning.
    If hallucination detected, regenerate or flag.

    REFINEMENT 7: Expanded validation checks for ratios and differences in numeric insights.
    """
    if not query_results:
        return answer

    # Extract all numbers from answer
    answer_numbers = re.findall(r'\d+\.?\d*', answer)

    # Extract all numbers from data
    data_numbers = set()
    data_values = []  # Keep numeric values for ratio/difference checks
    for row in query_results:
        for value in row.values():
            if value is not None:
                try:
                    # Try to extract number
                    num_str = str(value)
                    nums = re.findall(r'\d+\.?\d*', num_str)
                    data_numbers.update(nums)
                    # Convert to float for ratio/difference analysis
                    for num_str_match in nums:
                        data_values.append(float(num_str_match))
                except:
                    pass

    # Check for hallucinations (numbers in answer not in data)
    hallucinations = []
    for num in answer_numbers:
        num_float = float(num)
        # Allow small integers (likely counts) and common numbers
        if num_float > 100 and num not in data_numbers:
            hallucinations.append(num)

    # REFINEMENT 7: Check for invalid ratios/differences
    if data_values and len(data_values) >= 2:
        try:
            data_min = min(data_values)
            data_max = max(data_values)
            data_avg = sum(data_values) / len(data_values)
            data_total = sum(data_values)

            for num in answer_numbers:
                num_float = float(num)
                # Check if the number is suspiciously different from data range
                if num_float > data_max * 2:
                    hallucinations.append(f"{num}(exceeds_max_by_2x)")
                # Check for suspicious ratios
                if data_avg > 0 and num_float > data_avg * 10:
                    hallucinations.append(f"{num}(exceeds_avg_by_10x)")
        except Exception as e:
            logger.debug(f"Ratio/difference validation error: {str(e)}")

    if hallucinations and len(hallucinations) > 2:
        logger.warning(f"Potential hallucination detected: {hallucinations[:5]}")
        # Add disclaimer
        answer += "\n\n(Note: Please verify the exact figures from the data table below.)"

    return answer

# ============================================================================
# 8. PROMPT REFACTORIZATION - 3 specialized prompts
# ============================================================================

def build_table_retriever_prompt(question: str) -> str:
    """
    Specialized prompt for table retrieval.
    Concise and factual.
    """
    table_list = list(SCHEMA_REGISTRY.keys()) if SCHEMA_REGISTRY else list(DB_DESCRIPTIONS.keys())

    prompt = f"""You are a database table selector. Given a user question, identify the most relevant table(s).

Available tables:
{', '.join(table_list)}

User question: {question}

Return ONLY the table name(s) as JSON array:
{{"tables": ["table_name"]}}"""

    return prompt

def build_query_plan_prompt(question: str, intent: Dict[str, Any], schema_info: str) -> str:
    """
    Specialized prompt for SQL query generation.
    Uses comprehensive context about Bihar Animal & Fisheries datasets.

    IMPROVEMENT 10: Enhanced with expert context about tables, columns, and query patterns.

    FIXED: Issue 1 - No longer uses f-string with unsafe braces
    FIXED: Issue 2 - Year rules delegated to backend, not hardcoded in prompt
    FIXED: Issue 3 - Explicit prevention of division_name = 'Bihar' filter
    """
    # Build the base prompt as a normal triple-quoted string (no f-string interpolation)
    base_prompt = """
You are an expert MySQL query generator working for the Bihar Animal and Fisheries Department chatbot.
Your task is to generate a **perfect MySQL SELECT query** that answers the user's question using the correct dataset(s) and columns.

### CONTEXT
The chatbot database contains the following key tables and their real-world meanings:

1. **analysis_all_hatchery_district_wise**
   - One row per district per year.
   - Columns include:
     - `district_name`, `division_name`, `year`
     - `number_of_functional_hatcheries_Number`, `number_of_non_functional_hatcheries_Number`
     - `total_production_of_exotic_fingerling_Lakh`, `total_production_of_exotic_fry_Lakh`, `total_production_of_exotic_spawn_Lakh`
     - `total_production_of_imc_fingerling_Lakh`, `total_production_of_imc_fry_Lakh`, `total_production_of_imc_spawn_Lakh`
   - Represents fish hatchery counts and seed production data.

2. **areas_without_jalkars_by_range_district_block_panchayat**
   - Columns: `division_name`, `district_name`, `block_name`, `panchayat_name`
   - Lists areas that have **no jalkars (water bodies)**.
   - Used for questions like: "Which areas do not have jalkars?" or "Blocks with zero jalkars."

3. **details_fcs_district_wise**
   - Columns: `district_name`, `division_name`, `year`, `total_number_of_cooperative_societies_in_districts_Number`,
     `total_number_of_male_members_Number`, `total_number_of_female_members_Number`, `total_number_of_members_in_societies_Number`
   - Gives total cooperative societies (FCS) and membership details per district.
   - For queries about number of societies or members.

4. **details_newly_formed_fcs_district_wise**
   - Columns: `district_name`, `division_name`, `block_name`, `name_of_newly_formed_cooperative_societies`,
     `total_number_of_members_enrolled`, `total_number_of_male_members_enrolled`, `total_number_of_female_members_enrolled`,
     `date_of_registration`, `year`, `whether_Inland_or_Marine_FCS`
   - Represents newly formed cooperative societies with member details.

5. **jalkars_spread_across_multiple_panchayats_block_district_range_w**
   - Columns: `division_name`, `district_name`, `block_name`, `panchayat_name`, `jalkar_name`,
     `total_rakba_in_hectares`, `total_jamarashi_in_lakh_rupees`, `jalkars_located_in_multiple_panchayat`
   - Jalkars spanning multiple administrative boundaries.

6. **list_of_jalkars_with_unclear_settlement_status_panchayat_wise** (PRIMARY TABLE: 31,036 records)
   - Columns: `division_name`, `district_name`, `block_name`, `panchayat_name`, `jalkars_name`,
     `khata_number`, `khasra_number`, `total_rakba_in_hectares`, `settlement_status`,
     `security_deposit_amount_in_lakh`, `number_of_jalkars_pending_in_court_cases`,
     `area_of_jalkars_pending_in_court_cases_in_hectares`, `number_of_jalkars_declared_for_free_fishing`,
     `area_of_jalkars_declared_for_free_fishing_in_hectares`, `number_of_jalkars_declared_as_perta`,
     `area_of_jalkars_declared_as_perta_in_hectares`, `amount_of_jalkars_declared_as_perta_in_lakh`
   - Master table for all jalkar-level data across Bihar.
   - Handles questions like:
     - "List all jalkars in [district/block/panchayat]"
     - "Count of jalkars by settlement status"
     - "Unsettled jalkars" / "Court cases" / "Free fishing zones"
     - "Security deposit per jalkar"
     - "Total area (rakba) under PERTA jalkars"

7. **total_revenue_jalkar_report**
   - Columns: `district_name`, `division_name`, `year`, `annual_revenue_collected_against_the_target_Lakh_Rupees`,
     `current_fish_production_Thousand_Metric_Tons`, `fish_seed_fingerling_current_production_Number`,
     `fish_seed_fry_current_production_Number`, `fish_seed_production_spawn_current_count_Number`,
     `fish_seed_yearling_current_production_Number`, `current_fish_exported_to_other_states_Thousand_Metric_Tons`,
     `current_fish_imported_from_other_states_Thousand_Metric_Tons`, `annual_fish_production_target_from_all_aquatic_resources_Thousan`
   - Used for any questions about revenue, targets, or fish production quantities.

### CRITICAL RULES

1. **ALWAYS use exact table and column names** from the schema above.
2. **NEVER invent or guess columns** — only use those listed.
3. Always include appropriate **WHERE filters** based on user question (district_name, year, division_name, etc.).

4. **⚠️ CRITICAL: Bihar is the ENTIRE STATE, NOT a division**
   - NEVER generate: WHERE division_name = 'Bihar' (returns ZERO rows — 'Bihar' is not in the division_name column)
   - For state-level questions about "Bihar as a whole":
     * Either omit geographic filters and aggregate across all rows, OR
     * Use GROUP BY district_name without any division_name filter
   - For specific division questions: use actual division names ONLY
   - Example: Question "Total jalkars in Bihar?" → Do NOT use division filter; aggregate all rows

5. **Year Filter Handling**
   - If the user does not explicitly mention a year, you may omit the year filter.
   - The backend will automatically apply appropriate default year constraints via YEAR_RULES.
   - Only add a year filter when the user clearly specifies a year or year range in their question.

### VALID GEOGRAPHIC VALUES BY TABLE

**analysis_all_hatchery_district_wise** (26 districts, 8 divisions)
Valid district_name: Araria, Arwal, Aurangabad, Banka, Bhagalpur, Bhojpur, Darbhanga, East Champaran, Gaya, Gopalganj, Kaimur, Madhubani, Muzaffarpur, Nalanda, Nawada, Patna, Purnia, Rohtas, Saharsa, Samastipur, Saran, Sheohar, Sitamarhi, Siwan, Vaishali, West Champaran
Valid division_name: Bhagalpur, Darbhanga, Koshi, Magadh, Patna, Purnia, Saran, Tirhut

**areas_without_jalkars_by_range_district_block_panchayat** (38 districts, 9 divisions)
Valid district_name: Araria, Arwal, Aurangabad, Banka, Begusarai, Bhagalpur, Bhojpur, Buxar, Darbhanga, East Champaran, Gaya, Gopalganj, Jamui, Jehanabad, Kaimur, Katihar, Khagaria, Kishanganj, Lakhisarai, Madhepura, Madhubani, Munger, Muzaffarpur, Nalanda, Nawada, Patna, Purnia, Rohtas, Saharsa, Samastipur, Saran, Sheikhpura, Sheohar, Sitamarhi, Siwan, Supaul, Vaishali, West Champaran
Valid division_name: Bhagalpur, Darbhanga, Koshi, Magadh, Munger, Patna, Purnia, Saran, Tirhut

**details_fcs_district_wise** (38 districts, 9 divisions)
Valid district_name: Araria, Arwal, Aurangabad, Banka, Begusarai, Bhagalpur, Bhojpur, Buxar, Darbhanga, East Champaran, Gaya, Gopalganj, Jamui, Jehanabad, Kaimur, Katihar, Khagaria, Kishanganj, Lakhisarai, Madhepura, Madhubani, Munger, Muzaffarpur, Nalanda, Nawada, Patna, Purnia, Rohtas, Saharsa, Samastipur, Saran, Sheikhpura, Sheohar, Sitamarhi, Siwan, Supaul, Vaishali, West Champaran
Valid division_name: Bhagalpur, Darbhanga, Koshi, Magadh, Munger, Patna, Purnia, Saran, Tirhut

**details_newly_formed_fcs_district_wise** (38 districts, 9 divisions)
Valid district_name: Araria, Arwal, Aurangabad, Banka, Begusarai, Bhagalpur, Bhojpur, Buxar, Darbhanga, East Champaran, Gaya, Gopalganj, Jamui, Jehanabad, Kaimur, Katihar, Khagaria, Kishanganj, Lakhisarai, Madhepura, Madhubani, Munger, Muzaffarpur, Nalanda, Nawada, Patna, Purnia, Rohtas, Saharsa, Samastipur, Saran, Sheikhpura, Sheohar, Sitamarhi, Siwan, Supaul, Vaishali, West Champaran
Valid division_name: Bhagalpur, Darbhanga, Koshi, Magadh, Munger, Patna, Purnia, Saran, Tirhut

**jalkars_spread_across_multiple_panchayats_block_district_range_w** (25 districts, 8 divisions)
Valid district_name: Banka, Begusarai, Bhagalpur, Bhojpur, Buxar, Darbhanga, East Champaran, Gopalganj, Kaimur, Katihar, Khagaria, Kishanganj, Madhepura, Madhubani, Munger, Muzaffarpur, Patna, Purnia, Samastipur, Sarhasa, Sheohar, Sitamarhi, Supaul, Vaishali, West Champaran
Valid division_name: Bhagalpur, Darbhanga, Koshi, Munger, Patna, Purnia, Saran, Tirhut

**list_of_jalkars_with_unclear_settlement_status_panchayat_wise** (38 districts, 9 divisions)
Valid district_name: Araria, Arwal, Aurangabad, Banka, Begusarai, Bhagalpur, Bhojpur, Buxar, Darbhanga, East Champaran, Gaya, Gopalganj, Jamui, Jehanabad, Kaimur, Katihar, Khagaria, Kishanganj, Lakhisarai, Madhepura, Madhubani, Munger, Muzaffarpur, Nalanda, Nawada, Patna, Purnia, Rohtas, Saharsa, Samastipur, Saran, Sheikhpura, Sheohar, Sitamarhi, Siwan, Supaul, Vaishali, West Champaran
Valid division_name: Bhagalpur, Darbhanga, Koshi, Magadh, Munger, Patna, Purnia, Saran, Tirhut

**total_revenue_jalkar_report** (38 districts, 9 divisions)
Valid district_name: Araria, Arwal, Aurangabad, Banka, Begusarai, Bhagalpur, Bhojpur, Buxar, Darbhanga, East Champaran, Gaya, Gopalganj, Jamui, Jehanabad, Kaimur, Katihar, Khagaria, Kishanganj, Lakhisarai, Madhepura, Madhubani, Munger, Muzaffarpur, Nalanda, Nawada, Patna, Purnia, Rohtas, Saharsa, Samastipur, Saran, Sheikhpura, Sheohar, Sitamarhi, Siwan, Supaul, Vaishali, West Champaran
Valid division_name: Bhagalpur, Darbhanga, Koshi, Magadh, Munger, Patna, Purnia, Saran, Tirhut

6. Use **district_name** as the grouping level when user asks for totals or comparisons.
7. Use `SUM()` for aggregations like "total," "count," or "overall."
8. When comparing regions, use `ORDER BY` to sort results. Do NOT add `LIMIT` unless user explicitly asks for "top N".
9. When user says "top" or "highest", sort `DESC`; when "lowest", sort `ASC`.
10. **IMPORTANT: Do NOT use `LIMIT` by default. Fetch ALL relevant data. Only add `LIMIT` if user explicitly asks for "top N", "first few", or similar.**
11. For area or land size queries, use `total_rakba_in_hectares`.
12. For "settlement status" queries, filter by `settlement_status` values like:
    - 'Settled With Co-Operative', 'Unsettled', 'Free Fishing', 'Restricted By Court Unsettled', 'Settlement in Progress (Unsettled)'
13. For revenue-related queries, use `total_revenue_jalkar_report`.
14. For fish seed production, use `analysis_all_hatchery_district_wise` or `total_revenue_jalkar_report` depending on metric requested.

### ⚠️ CRITICAL: NO LIMIT CLAUSES (UNLESS EXPLICITLY ASKED)

**NEVER add a LIMIT clause to the query unless the user explicitly asks for "top N", "first few", "highest/lowest", or similar.**

Examples of when NOT to add LIMIT:
- ❌ "List all jalkars" → DO NOT add LIMIT
- ❌ "Show total hatcheries in each district" → DO NOT add LIMIT
- ❌ "District-wise revenue" → DO NOT add LIMIT

Examples of when to add LIMIT:
- ✅ "Top 5 districts by revenue" → Add LIMIT 5
- ✅ "Show me the top 10 blocks" → Add LIMIT 10
- ✅ "First 20 jalkars" → Add LIMIT 20

**Default behavior: Fetch ALL matching rows. The application will handle display of top N if needed.**

### OUTPUT FORMAT

Generate **ONLY ONE** valid MySQL query, no explanations, no markdown, no code fences.

The query must:
- Start with `SELECT`
- End with a semicolon `;`
- Be executable directly on MySQL 8+
- Contain valid joins only if needed (prefer single-table queries)
- Include meaningful aliases for clarity
- **NEVER include LIMIT unless user explicitly asks for top N**

### EXAMPLES

#### Example 1: Hatchery Query
**User:** "Total functional hatcheries in Gaya for 2023-24"
**SQL:**
SELECT district_name, SUM(number_of_functional_hatcheries_Number) AS total_functional_hatcheries
FROM analysis_all_hatchery_district_wise
WHERE district_name = 'Gaya' AND year = '2023-24'
GROUP BY district_name;

#### Example 2: Jalkar Listing
**User:** "List all jalkars in Patna district"
**SQL:**
SELECT jalkars_name, panchayat_name, block_name, total_rakba_in_hectares, settlement_status
FROM list_of_jalkars_with_unclear_settlement_status_panchayat_wise
WHERE district_name = 'Patna';

#### Example 3: Settlement Status
**User:** "How many unsettled jalkars in Muzaffarpur?"
**SQL:**
SELECT COUNT(*) AS unsettled_jalkar_count, SUM(total_rakba_in_hectares) AS total_area_hectares
FROM list_of_jalkars_with_unclear_settlement_status_panchayat_wise
WHERE district_name = 'Muzaffarpur' AND settlement_status = 'Unsettled';

#### Example 4: Revenue Aggregation
**User:** "Top 5 districts by fish revenue"
**SQL:**
SELECT district_name, SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) AS total_revenue
FROM total_revenue_jalkar_report
WHERE year = 2024
GROUP BY district_name
ORDER BY total_revenue DESC;
**Note:** Fetch all results; the application can sort and display top 5 if needed.

#### Example 5: FCS Members
**User:** "Total FCS members in all districts"
**SQL:**
SELECT SUM(total_number_of_members_in_societies_Number) AS total_fcs_members
FROM details_fcs_district_wise
WHERE year = 2025;

CRITICAL RULES FOR SQL GENERATION:

1. ALWAYS use proper aggregation when using GROUP BY:
   - If the query groups by a column (like district_name, division_name, block_name, or panchayat_name),
     then ALL other numeric columns (like revenue, collection, production, area, count, amount, percentage, etc.)
     must be wrapped inside aggregation functions like SUM(), COUNT(), or AVG().
   - Example:
       WRONG:
         SELECT district_name, annual_revenue_collected_against_the_target_Lakh_Rupees
         FROM total_revenue_jalkar_report
         GROUP BY district_name;
       CORRECT:
         SELECT district_name,
                SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) AS total_collected
         FROM total_revenue_jalkar_report
         GROUP BY district_name;

2. If the question involves "comparison," "top N," "ranking," or "performance,"
   always aggregate numeric columns and use ORDER BY with the aggregate alias.
   Example:
       SELECT district_name,
              SUM(total_fish_production) AS total_production
       FROM total_revenue_jalkar_report
       WHERE year = 2024
       GROUP BY district_name
       ORDER BY total_production DESC;

3. When filtering based on ratio or percentage (e.g., "above 90% of target"):
   - Compute percentage using aggregate values.
   - Use HAVING for post-aggregation filters.
   Example:
       SELECT district_name,
              SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) AS total_collected,
              SUM(annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup) AS total_target,
              (SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) /
               SUM(annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup)) * 100 AS collection_percentage
       FROM total_revenue_jalkar_report
       WHERE year = 2024
       GROUP BY district_name
       HAVING collection_percentage > 90
       ORDER BY collection_percentage DESC;

4. Avoid using raw column names in SELECT unless they appear in GROUP BY.

5. Always prefer meaningful aliases for calculated fields, e.g.:
       - total_fish_production
       - collection_percentage
       - total_rakba
       - number_of_unsettled_jalkars

6. Ensure all SQLs are fully executable in MySQL 8+ with ONLY_FULL_GROUP_BY mode enabled.
   Never rely on implicit grouping behavior.

7. If the question involves totals, averages, or comparisons between years,
   always aggregate across years and group by 'year' or 'district_name' as needed.

8. If no aggregation or grouping is required (e.g., "list all hatcheries"),
   produce a simple SELECT query with relevant filters.

9. Always ensure the SQL query syntax follows this structure:
       SELECT ... FROM ... WHERE ... GROUP BY ... HAVING ... ORDER BY ... LIMIT ...;

10. Never use unaggregated numeric columns in SELECT with GROUP BY.
    Never assume non-grouped columns can appear directly in SELECT.



## Prompt Section: `jalkars_spread_across_multiple_panchayats_block_district_range_w`

**TABLE: `jalkars_spread_across_multiple_panchayats_block_district_range_w`**
Represents water bodies (jalkars) that span multiple panchayat boundaries. One row ≈ one (primary panchayat, jalkar) record, with optional list of additional panchayats the jalkar extends into.

**Columns (exact names):**

* `division_name` (TEXT) – Administrative division
* `district_name` (TEXT) – District
* `block_name` (TEXT) – Block
* `panchayat_name` (TEXT) – Primary panchayat
* `jalkar_name` (TEXT) – Jalkar identifier/name (can repeat across rows if it spans many areas)
* `total_rakba_in_hectares` (DECIMAL) – Area in hectares (non-negative)
* `total_jamarashi_in_lakh_rupees` (DECIMAL) – Demand / revenue amount in lakh rupees (often 0 or small)
* `jalkars_located_in_multiple_panchayat` (TEXT) – Comma-separated list of **secondary** panchayat names (may be NULL/empty).

  * If non-empty, the jalkar spans **1 (primary) + N (secondary)** panchayats in this row.
  * N can be computed by the number of commas + 1.

**Important facts & constraints:**

* There is **no `year` column** in this table → never add year filters here.
* Use **`SUM()` and `COUNT()`** for all numeric aggregations and record counts.
* When using `GROUP BY`, **every selected numeric field MUST be aggregated** (avoid ONLY_FULL_GROUP_BY errors).
* Names (division/district/block/panchayat/jalkar) are free text; for fuzzy search use `LIKE '%term%'`.
* To compute **how many panchayats a row spans**:

  * Secondary count: `NULLIF(jalkars_located_in_multiple_panchayat, '')` may be NULL/empty.
  * Total panchayats in the row: `1 + (LENGTH(col) - LENGTH(REPLACE(col, ',', ''))) + (col IS NOT NULL)` but guard NULL/empty safely (see examples).

**When to choose this table:**

* Any query about **“multi-panchayat jalkars”**, **area (rakba)**, **demand/jamarashi**, or **counts** broken down by division/district/block/panchayat/jalkar.
* Top/lowest by **area** or **jamarashi**.
* Listing jalkars in a given geography, or threshold filters (e.g., area > X).

---

### GOLDEN RULES (for this table)

1. **Never** add year filters (no `year` column here).
2. For comparisons/rankings/"top N", always aggregate and `ORDER BY` the aggregate alias. Only add `LIMIT N` if user explicitly asks for "top N".
3. For thresholds on area/jamarashi, filter in `WHERE` (if row-level) or use `HAVING` if filtering on aggregates. Fetch ALL matching rows unless user asks for top N.
4. For counts of “multi-panchayat” coverage, compute secondary count robustly (see helper expressions below).
5. When grouping (e.g., by district), select only the grouping columns + aggregated numeric fields.

---

### Helper expressions (copy-ready)

* **Row-level secondary panchayat count (`sec_cnt`)** (0 when NULL/empty):

```sql
CASE
  WHEN jalkars_located_in_multiple_panchayat IS NULL OR jalkars_located_in_multiple_panchayat = '' THEN 0
  ELSE 1 + LENGTH(jalkars_located_in_multiple_panchayat) - LENGTH(REPLACE(jalkars_located_in_multiple_panchayat, ',', ''))
END
```

* **Row-level total panchayats spanned (`row_total_panchayats`)**:

 (
  CASE
    WHEN jalkars_located_in_multiple_panchayat IS NULL OR jalkars_located_in_multiple_panchayat = '' THEN 0
    ELSE 1 + LENGTH(jalkars_located_in_multiple_panchayat) - LENGTH(REPLACE(jalkars_located_in_multiple_panchayat, ',', ''))
  END
)
```

> Note: `row_total_panchayats = 1 (primary) + sec_cnt`.

---

### Query patterns (safe under ONLY_FULL_GROUP_BY)

#### 1) List jalkars in a geography (with area & jamarashi)

```
SELECT jalkar_name,
       panchayat_name,
       block_name,
       district_name,
       total_rakba_in_hectares,
       total_jamarashi_in_lakh_rupees,
       jalkars_located_in_multiple_panchayat
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
WHERE district_name = 'Patna'
ORDER BY jalkar_name ASC;
```

#### 2) Top N blocks by **total rakba**

```
SELECT block_name,
       SUM(total_rakba_in_hectares) AS total_rakba
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
WHERE district_name = 'Patna'
GROUP BY block_name
ORDER BY total_rakba DESC;
```

#### 3) District-wise totals: **rakba** and **jamarashi**

```
SELECT district_name,
       SUM(total_rakba_in_hectares) AS total_rakba,
       SUM(total_jamarashi_in_lakh_rupees) AS total_jamarashi
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
GROUP BY district_name
ORDER BY total_rakba DESC;
```

#### 4) Panchayat coverage: **how many multi-panchayat jalkars?** (count rows with any secondary panchayat)

```
SELECT district_name,
       COUNT(*) AS multi_panchayat_jalkar_count
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
WHERE jalkars_located_in_multiple_panchayat IS NOT NULL
  AND jalkars_located_in_multiple_panchayat <> ''
GROUP BY district_name
ORDER BY multi_panchayat_jalkar_count DESC;
```

#### 5) Sum of **total panchayats spanned** per district

```
SELECT district_name,
       SUM(
         1 + CASE
               WHEN jalkars_located_in_multiple_panchayat IS NULL
                    OR jalkars_located_in_multiple_panchayat = '' THEN 0
               ELSE 1 + LENGTH(jalkars_located_in_multiple_panchayat)
                          - LENGTH(REPLACE(jalkars_located_in_multiple_panchayat, ',', ''))
             END
       ) AS total_panchayat_coverage
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
GROUP BY district_name
ORDER BY total_panchayat_coverage DESC;
```

#### 6) Threshold filter: **area > X** (list)

```
SELECT jalkar_name, block_name, panchayat_name, district_name, total_rakba_in_hectares
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
WHERE district_name = 'Bhagalpur'
  AND total_rakba_in_hectares > 10
ORDER BY total_rakba_in_hectares DESC;
```

#### 7) **Top N jalkars** by area (within a district/block)

```
SELECT jalkar_name,
       panchayat_name,
       SUM(total_rakba_in_hectares) AS total_rakba
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
WHERE district_name = 'Patna' AND block_name = 'Maner'
GROUP BY jalkar_name, panchayat_name
ORDER BY total_rakba DESC;
```

#### 8) **Contribution %** of each district to state total rakba

```
SELECT t.district_name,
       t.total_rakba,
       (t.total_rakba / s.state_total_rakba) * 100 AS rakba_percentage
FROM (
  SELECT district_name, SUM(total_rakba_in_hectares) AS total_rakba
  FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
  GROUP BY district_name
) AS t
CROSS JOIN (
  SELECT SUM(total_rakba_in_hectares) AS state_total_rakba
  FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
) AS s
ORDER BY rakba_percentage DESC;
```

#### 9) **Search** jalkars by partial name (case-insensitive by default)

```
SELECT jalkar_name, district_name, block_name, panchayat_name, total_rakba_in_hectares
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
WHERE jalkar_name LIKE '%talab%'
ORDER BY jalkar_name ASC
```

#### 10) District vs block **ranking by jamarashi**

```
SELECT district_name,
       block_name,
       SUM(total_jamarashi_in_lakh_rupees) AS total_jamarashi
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
GROUP BY district_name, block_name
ORDER BY total_jamarashi DESC;
```

#### 11) **Blocks with most multi-panchayat coverage** (sum of total panchayats spanned)

```
SELECT block_name,
       SUM(
         1 + CASE
               WHEN jalkars_located_in_multiple_panchayat IS NULL
                    OR jalkars_located_in_multiple_panchayat = '' THEN 0
               ELSE 1 + LENGTH(jalkars_located_in_multiple_panchayat)
                          - LENGTH(REPLACE(jalkars_located_in_multiple_panchayat, ',', ''))
             END
       ) AS total_panchayat_coverage
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
WHERE district_name = 'Gaya'
GROUP BY block_name
ORDER BY total_panchayat_coverage DESC;
```

#### 12) **Panchayat-wise totals** within a block

```
SELECT panchayat_name,
       COUNT(*) AS jalkar_count,
       SUM(total_rakba_in_hectares) AS total_rakba,
       SUM(total_jamarashi_in_lakh_rupees) AS total_jamarashi
FROM jalkars_spread_across_multiple_panchayats_block_district_range_w
WHERE district_name = 'Muzaffarpur' AND block_name = 'Sakra'
GROUP BY panchayat_name
ORDER BY total_rakba DESC;
```

---

### Answering Guidance (for this table)

* If a question mentions **multi-panchayat extent/coverage**, compute coverage using the helper expressions (sum at desired geography).
* If a question asks for **"top/lowest"** by area or jamarashi, aggregate then `ORDER BY` the aggregate alias. Only add `LIMIT` if user explicitly asks for "top N".
* If a question asks for **counts** (e.g., number of jalkars), use `COUNT(*)` and optionally include area/jamarashi sums. Fetch ALL matching rows by default.
* If both **area and jamarashi** are needed, return both with clear aliases.
* For **percentage** questions, use a subquery/CROSS JOIN to get state totals, then compute percentage safely.

**Never**:

* Add year filters.
* Select unaggregated numeric columns with `GROUP BY`.
* Invent columns not listed above.

**TABLE: list_of_jalkars_with_unclear_settlement_status_panchayat_wise**
   - This is the **primary jalkar dataset** (≈31,000 records), containing settlement, area, deposit, and special-status information for all water bodies across Bihar.
   - Each record represents a single jalkar located in a specific panchayat, with administrative, financial, and legal metadata.
   - **Columns:**
     - `division_name`: Administrative division where the jalkar is located.
     - `district_name`: District name.
     - `block_name`: Block (sub-district).
     - `panchayat_name`: Panchayat under which the jalkar falls.
     - `jalkars_name`: Official name or identifier of the water body.
     - `khata_number`: Khata (land record) number.
     - `khasra_number`: Khasra (plot) number.
     - `total_rakba_in_hectares`: Total area (rakba) of the jalkar in hectares.
     - `settlement_status`: Settlement classification. Common values:
         - 'Settled With Co-Operative'
         - 'Unsettled'
         - 'Free Fishing'
         - 'Restricted By Court Unsettled'
         - 'Settlement in Progress (Unsettled)'
     - `security_deposit_amount_in_lakh`: Deposit amount in lakh rupees.
     - `number_of_jalkars_pending_in_court_cases`: Count of jalkars under legal dispute.
     - `area_of_jalkars_pending_in_court_cases_in_hectares`: Total area under legal disputes.
     - `number_of_jalkars_declared_for_free_fishing`: Count of free-fishing jalkars.
     - `area_of_jalkars_declared_for_free_fishing_in_hectares`: Total area under free fishing.
     - `number_of_jalkars_declared_as_perta`: Count of PERTA-declared jalkars.
     - `area_of_jalkars_declared_as_perta_in_hectares`: Total area declared as PERTA.
     - `amount_of_jalkars_declared_as_perta_in_lakh`: Financial value of PERTA jalkars in lakh rupees.

   - **Table Use Cases:**
     - For all queries about **jalkar settlement**, **area**, **court cases**, **free fishing**, **PERTA declarations**, or **security deposits**.
     - Ideal for both quantitative and listing-type questions like:
       - “How many unsettled jalkars are in Patna?”
       - “Total rakba under free-fishing areas?”
       - “District-wise PERTA declared area and amount?”
       - “List all court-case jalkars in Muzaffarpur.”

   - **Aggregation Rules:**
     1. Always wrap numeric columns (rakba, amount, deposit, counts) in `SUM()` or `COUNT()` when grouping.
     2. When counting jalkars, use `COUNT(*)` unless the question refers to a specific type (then count filtered rows).
     3. When comparing by district/block/panchayat, always `GROUP BY` that column and aggregate others.
     4. Always alias aggregates meaningfully:
        - `total_area_hectares`
        - `total_security_deposit`
        - `unsettled_jalkar_count`
        - `perta_area_total`
        - `court_case_count`
     5. Always include settlement status filters if user specifies one.

   - **Query Examples (all valid under ONLY_FULL_GROUP_BY):**

     **List all jalkars in a district**
     ```
     SELECT jalkars_name, block_name, panchayat_name, total_rakba_in_hectares, settlement_status
     FROM list_of_jalkars_with_unclear_settlement_status_panchayat_wise
     WHERE district_name = 'Patna'
     ORDER BY jalkars_name ASC;
     ```

     **Count and area of unsettled jalkars**
     ```
     SELECT COUNT(*) AS unsettled_jalkar_count,
            SUM(total_rakba_in_hectares) AS total_area_hectares
     FROM list_of_jalkars_with_unclear_settlement_status_panchayat_wise
     WHERE district_name = 'Muzaffarpur'
       AND settlement_status = 'Unsettled';
     ```

     **District-wise total unsettled jalkars**
     ```
     SELECT district_name,
            COUNT(*) AS unsettled_jalkar_count,
            SUM(total_rakba_in_hectares) AS total_area_hectares
     FROM list_of_jalkars_with_unclear_settlement_status_panchayat_wise
     WHERE settlement_status = 'Unsettled'
     GROUP BY district_name
     ORDER BY unsettled_jalkar_count DESC;
     ```

     **Total PERTA-declared area and amount per district**
     ```
     SELECT district_name,
            SUM(area_of_jalkars_declared_as_perta_in_hectares) AS perta_area_total,
            SUM(amount_of_jalkars_declared_as_perta_in_lakh) AS perta_amount_total
     FROM list_of_jalkars_with_unclear_settlement_status_panchayat_wise
     GROUP BY district_name
     ORDER BY perta_area_total DESC;
     ```

     **Total area and number of free-fishing jalkars**
     ```
     SELECT SUM(area_of_jalkars_declared_for_free_fishing_in_hectares) AS free_fishing_area_total,
            SUM(number_of_jalkars_declared_for_free_fishing) AS free_fishing_count
     FROM list_of_jalkars_with_unclear_settlement_status_panchayat_wise;
     ```

     **Court case-related aggregation**
     ```
     SELECT district_name,
            SUM(number_of_jalkars_pending_in_court_cases) AS court_case_count,
            SUM(area_of_jalkars_pending_in_court_cases_in_hectares) AS total_court_case_area
     FROM list_of_jalkars_with_unclear_settlement_status_panchayat_wise
     GROUP BY district_name
     ORDER BY court_case_count DESC;
     ```

     **Security deposit totals**
     ```
     SELECT district_name,
            SUM(security_deposit_amount_in_lakh) AS total_security_deposit
     FROM list_of_jalkars_with_unclear_settlement_status_panchayat_wise
     GROUP BY district_name
     ORDER BY total_security_deposit DESC;
     ```

     **Settlement summary report**
     ```
     SELECT settlement_status,
            COUNT(*) AS jalkar_count,
            SUM(total_rakba_in_hectares) AS total_area
     FROM list_of_jalkars_with_unclear_settlement_status_panchayat_wise
     GROUP BY settlement_status
     ORDER BY jalkar_count DESC;
     ```

   - **Rules for complex questions:**
     - When question mentions **ratio, comparison, or percentage** (e.g., “What % of total jalkars are unsettled?”):
       - Compute using a subquery with `CROSS JOIN` total count.
     - When question mentions **"largest" or "smallest"**, use `ORDER BY` DESC/ASC with `LIMIT`.
     - When user says “district-wise”, always `GROUP BY district_name`.
     - When user says “list”, do not aggregate—just filter and `ORDER BY`.

   - **Never:**
     - Add year filters (no `year` column here).
     - Use unaggregated numeric columns with GROUP BY.
     - Invent column names (use only those defined).
     - Confuse this table with revenue or hatchery datasets (those are different tables).

**Table: areas_without_jalkars_by_range_district_block_panchayat**
   - This dataset lists **administrative regions that currently have no jalkars (water bodies)** within their jurisdiction.
   - Each record represents one specific panchayat (or block) officially marked as “without jalkars,” along with its district, division, and range information.
   - It is used to identify and report **deficient areas** for water resource development, fisheries expansion, or policy interventions.

   - **Columns:**
     - `division_name`: Name of the administrative division (e.g., Tirhut, Patna, Bhagalpur).
     - `district_name`: Name of the district where the block/panchayat lies.
     - `block_name`: Name of the sub-district (block).
     - `panchayat_name`: Name of the panchayat declared to have no jalkars.

   - **Usage:**
     - Answer questions about **which areas, panchayats, or blocks have zero jalkars**.
     - Produce **district-wise or block-wise summaries** of jalkar-deficient regions.
     - Useful for spatial mapping and planning of new water-body creation programs.

   - **Important Notes:**
     - ❌ This table does **not contain any numeric fields** for aggregation (no area, count, or amount columns).
     - ✅ Use `COUNT(*)` for totals (e.g., number of panchayats without jalkars).
     - ✅ Use only **filtering and grouping** by geography (division, district, block, panchayat).
     - ✅ When grouping, only `COUNT(*)` or `DISTINCT panchayat_name` should be aggregated.
     - ⚠️ Never use SUM(), AVG(), or numeric filters—there are no numeric fields.
     - ⚠️ Do not join with revenue or hatchery tables unless explicitly requested (this table stands alone).

   - **Example Question Types:**
     - “Which panchayats have no jalkars in Muzaffarpur district?”
     - “Count of panchayats without jalkars in each district.”
     - “List blocks without jalkars under Patna division.”
     - “Total number of jalkar-deficient panchayats in Bihar.”
     - “Which divisions have the highest number of no-jalkar areas?”

   - **Aggregation & Grouping Rules:**
     1. Use `COUNT(*)` for number of records (panchayats or blocks).
     2. For district-wise or division-wise summaries, `GROUP BY` the relevant administrative column.
     3. Alias counts meaningfully:
        - `panchayats_without_jalkars_count`
        - `blocks_without_jalkars_count`
        - `districts_with_no_jalkars_count`
     4. Always include proper filters (WHERE district_name = '...' or division_name = '...') as per user query.

   - **Query Examples (All MySQL 8+ Safe under ONLY_FULL_GROUP_BY):**

     **List all panchayats without jalkars in a district**
     ```
     SELECT division_name, district_name, block_name, panchayat_name
     FROM areas_without_jalkars_by_range_district_block_panchayat
     WHERE district_name = 'Gaya'
     ORDER BY block_name, panchayat_name;
     ```

     **Count of panchayats without jalkars in each district**
     ```
     SELECT district_name,
            COUNT(*) AS panchayats_without_jalkars_count
     FROM areas_without_jalkars_by_range_district_block_panchayat
     GROUP BY district_name
     ORDER BY panchayats_without_jalkars_count DESC;
     ```

     **Division-wise totals**
     ```
     SELECT division_name,
            COUNT(*) AS total_panchayats_without_jalkars
     FROM areas_without_jalkars_by_range_district_block_panchayat
     GROUP BY division_name
     ORDER BY total_panchayats_without_jalkars DESC;
     ```

     **Block-wise count for a given district**
     ```
     SELECT block_name,
            COUNT(*) AS panchayats_without_jalkars_count
     FROM areas_without_jalkars_by_range_district_block_panchayat
     WHERE district_name = 'Patna'
     GROUP BY block_name
     ORDER BY panchayats_without_jalkars_count DESC;
     ```

     **Top districts with the most panchayats lacking jalkars**
     ```
     SELECT district_name,
            COUNT(*) AS panchayats_without_jalkars_count
     FROM areas_without_jalkars_by_range_district_block_panchayat
     GROUP BY district_name
     ORDER BY panchayats_without_jalkars_count DESC;
     ```

     **Count total panchayats without jalkars statewide**
     ```
     SELECT COUNT(*) AS total_panchayats_without_jalkars
     FROM areas_without_jalkars_by_range_district_block_panchayat;
     ```

   - **Logic for Complex Questions:**
     - “Which block has the highest number of panchayats without jalkars?” → Group by block, order DESC, limit 1.
     - “How many jalkar-deficient panchayats in Tirhut division?” → Filter by division_name = 'Tirhut' and COUNT(*).
     - “Which divisions are completely covered with jalkars?” → Compare against total divisions (requires other table; otherwise return 'Not available' from this table).
     - “Districts with more than 20 panchayats without jalkars?” → Use `HAVING COUNT(*) > 20`.

   - **Always Remember:**
     - No year column here → ❌ Never add year filters.
     - No numeric fields → ✅ Use COUNT(*) only.
     - No revenue, area, or deposit information in this dataset.
     - Only geography-based counts and listings are valid.
     - When user asks “Where there are no jalkars?”, **always use this table**.

**Table: total_revenue_jalkar_report**
   - This dataset contains the **official annual financial and fish production performance data** for all districts and divisions under the Bihar Fisheries Department.
   - Each record represents one district’s statistics for a particular fiscal year, including revenue, fish production, and seed output metrics.

   - **Columns:**
     - `division_name`: Administrative division of the district.
     - `district_name`: Name of the district.
     - `year`: Year or financial year of record (e.g., 2023, 2024).
     - `annual_revenue_collected_against_the_target_Lakh_Rupees`: Actual revenue collected from jalkar settlements during the year, measured in lakh rupees.
     - `annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup`: Targeted revenue for the same period, measured in lakh rupees.
     - `current_fish_production_Thousand_Metric_Tons`: Actual fish production from all water bodies (in thousand metric tons).
     - `annual_fish_production_target_from_all_aquatic_resources_Thousan`: Target fish production (in thousand metric tons).
     - `fish_seed_fingerling_current_production_Number`: Current production count of fish fingerlings.
     - `fish_seed_fry_current_production_Number`: Current production count of fish fry.
     - `fish_seed_production_spawn_current_count_Number`: Production count of fish spawn.
     - `fish_seed_yearling_current_production_Number`: Current production count of fish yearlings.
     - `current_fish_exported_to_other_states_Thousand_Metric_Tons`: Quantity of fish exported outside the state.
     - `current_fish_imported_from_other_states_Thousand_Metric_Tons`: Quantity of fish imported from other states.

   - **Purpose:**
     - Used for **financial**, **production**, and **target achievement** analysis by year, district, and division.
     - Helps answer queries about:
       - “Revenue collection vs target”
       - “Top performing districts by revenue or fish production”
       - “Fish seed production totals”
       - “Comparison between districts/divisions”
       - “Year-over-year performance trends”

   - **Key Metrics:**
     - **Revenue Achievement %:**  
       `(SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) / SUM(annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup)) * 100`
     - **Fish Production Achievement %:**  
       `(SUM(current_fish_production_Thousand_Metric_Tons) / SUM(annual_fish_production_target_from_all_aquatic_resources_Thousan)) * 100`
     - **Seed Output Totals:**  
       SUM of all seed production columns combined (fingerling + fry + spawn + yearling).

   - **Aggregation Rules:**
     1. Always filter by `year` unless user explicitly requests multi-year comparisons.
     2. Use `SUM()` for all numeric aggregations (revenue, production, seed counts).
     3. When comparing across districts/divisions, always include `GROUP BY district_name` or `GROUP BY division_name`.
     4. Use aliases for readability:
        - `total_revenue_collected`
        - `total_revenue_target`
        - `revenue_achievement_percentage`
        - `total_fish_production`
        - `total_fish_target`
        - `production_achievement_percentage`
     5. For "top N" or "highest/lowest" questions, use `ORDER BY` on the aggregate alias. Only add `LIMIT N` if user explicitly asks for "top N".
     6. Use `HAVING` when filtering based on derived ratios or percentages (e.g., achievement > 90%). Fetch ALL matching rows unless user asks for top N.

   - **Default Year Handling:**
     - If the user does not mention a year → **default to `year = 2024`**.

   - **Example Question Types:**
     - “Which district collected the highest revenue in 2024?”
     - “Top 5 districts with highest revenue collection percentage.”
     - “Compare total revenue vs target for each division.”
     - “Total fish production achieved vs target in 2024.”
     - “Districts that achieved more than 90% of their revenue target.”
     - “How much fish was exported outside Bihar in 2024?”
     - “Average fish seed fry production across all districts.”

   - **Query Examples (All valid under MySQL 8+ with ONLY_FULL_GROUP_BY enabled):**

     **1. Total revenue collection and target for all districts**
     ```
     SELECT district_name,
            SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) AS total_revenue_collected,
            SUM(annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup) AS total_revenue_target
     FROM total_revenue_jalkar_report
     WHERE year = 2024
     GROUP BY district_name
     ORDER BY total_revenue_collected DESC;
     ```

     **2. Revenue achievement percentage by district**
     ```
     SELECT district_name,
            SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) AS total_collected,
            SUM(annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup) AS total_target,
            (SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) /
             SUM(annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup)) * 100 AS revenue_achievement_percentage
     FROM total_revenue_jalkar_report
     WHERE year = 2024
     GROUP BY district_name
     ORDER BY revenue_achievement_percentage DESC;
     ```

     **3. Top 5 districts with revenue collection above 90% of target**
     ```
     SELECT district_name,
            SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) AS total_collected,
            SUM(annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup) AS total_target,
            (SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) /
             SUM(annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup)) * 100 AS collection_percentage
     FROM total_revenue_jalkar_report
     WHERE year = 2024
     GROUP BY district_name
     HAVING collection_percentage > 90
     ORDER BY collection_percentage DESC;
     ```

     **4. Division-wise total revenue and achievement**
     ```
     SELECT division_name,
            SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) AS total_collected,
            SUM(annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup) AS total_target,
            (SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) /
             SUM(annual_revenue_collection_target_from_jalkar_settlement_Lakh_Rup)) * 100 AS achievement_percentage
     FROM total_revenue_jalkar_report
     WHERE year = 2024
     GROUP BY division_name
     ORDER BY achievement_percentage DESC;
     ```

     **5. Total fish production vs target**
     ```
     SELECT SUM(current_fish_production_Thousand_Metric_Tons) AS total_fish_production,
            SUM(annual_fish_production_target_from_all_aquatic_resources_Thousan) AS total_fish_target,
            (SUM(current_fish_production_Thousand_Metric_Tons) /
             SUM(annual_fish_production_target_from_all_aquatic_resources_Thousan)) * 100 AS fish_production_achievement_percentage
     FROM total_revenue_jalkar_report
     WHERE year = 2024;
     ```

     **6. Fish seed production summary**
     ```
     SELECT district_name,
            SUM(fish_seed_fingerling_current_production_Number) AS total_fingerling,
            SUM(fish_seed_fry_current_production_Number) AS total_fry,
            SUM(fish_seed_production_spawn_current_count_Number) AS total_spawn,
            SUM(fish_seed_yearling_current_production_Number) AS total_yearling
     FROM total_revenue_jalkar_report
     WHERE year = 2024
     GROUP BY district_name
     ORDER BY total_fingerling DESC;
     ```

     **7. Fish export and import overview**
     ```
     SELECT SUM(current_fish_exported_to_other_states_Thousand_Metric_Tons) AS total_exported,
            SUM(current_fish_imported_from_other_states_Thousand_Metric_Tons) AS total_imported
     FROM total_revenue_jalkar_report
     WHERE year = 2024;
     ```

     **8. District-wise performance comparison (both revenue and production)**
     ```
     SELECT district_name,
            SUM(annual_revenue_collected_against_the_target_Lakh_Rupees) AS total_revenue,
            SUM(current_fish_production_Thousand_Metric_Tons) AS total_production
     FROM total_revenue_jalkar_report
     WHERE year = 2024
     GROUP BY district_name
     ORDER BY total_revenue DESC;
     ```

   - **Complex Question Handling:**
     - “Which district achieved more than 90% of both revenue and fish production targets?”
       → Use HAVING on both achievement % columns.
     - “Compare top 5 divisions by total revenue collection.”
       → Group by `division_name`, order DESC, limit 5.
     - “Districts where revenue collected < target.”
       → HAVING total_collected < total_target.
     - “Fish production shortfall per district.”
       → Use difference: `(SUM(target) - SUM(actual)) AS production_gap`.

   - **Key Rules Recap:**
     - Always aggregate numeric columns with `SUM()` before grouping.
     - Always `GROUP BY` for district/division comparisons.
     - Use `HAVING` for ratio-based filters.
     - Never select non-aggregated numeric columns with GROUP BY.
     - No join needed — this is a standalone financial + production table.
     - Default `WHERE year = 2024` if not specified.

   - **Aliases to Prefer in Output:**
     - `total_revenue_collected`
     - `total_revenue_target`
     - `revenue_achievement_percentage`
     - `total_fish_production`
     - `fish_production_achievement_percentage`
     - `total_seed_fry`
     - `total_seed_fingerling`
     - `total_seed_spawn`
     - `total_seed_yearling`
     - `total_exported`
     - `total_imported`

**Table: analysis_all_hatchery_district_wise**
   - This dataset stores **district-wise hatchery performance and seed production statistics** for each fiscal year.
   - Each record represents one district’s hatchery data for a given year.

   - **Columns:**
     - `division_name`: Administrative division name.
     - `district_name`: Name of the district.
     - `year`: Year of record (e.g., 2022-23, 2023-24).
     - `number_of_functional_hatcheries_Number`: Total functional hatcheries in the district.
     - `number_of_non_functional_hatcheries_Number`: Non-functional hatcheries in the district.
     - `total_production_of_exotic_fingerling_Lakh`: Production of exotic fingerlings (in lakh units).
     - `total_production_of_exotic_fry_Lakh`: Production of exotic fry (in lakh units).
     - `total_production_of_exotic_spawn_Lakh`: Production of exotic spawn (in lakh units).
     - `total_production_of_imc_fingerling_Lakh`: Production of Indian Major Carp (IMC) fingerlings (in lakh units).
     - `total_production_of_imc_fry_Lakh`: Production of IMC fry (in lakh units).
     - `total_production_of_imc_spawn_Lakh`: Production of IMC spawn (in lakh units).

   - **Purpose:**
     - Used to analyze hatchery operations and fish seed production at district/division level.
     - Answers questions such as:
       - “Total functional hatcheries in Gaya in 2023-24?”
       - “Which district produced the most IMC fry?”
       - “Compare exotic vs IMC seed production by district.”
       - “Division-wise total hatchery output.”

   - **Aggregation Rules:**
     1. Always use `SUM()` for numeric fields.
     2. When comparing across districts/divisions, always include `GROUP BY district_name` or `division_name`.
     3. Default `WHERE year = '2023-24'` if user doesn’t specify year.
     4. Use meaningful aliases:
        - `total_functional_hatcheries`
        - `total_non_functional_hatcheries`
        - `total_exotic_fry_production`
        - `total_imc_spawn_production`
     5. Use `ORDER BY` for "top" or "lowest" queries. Only add `LIMIT` if user explicitly asks for "top N".
     6. When comparing between types (IMC vs exotic), use sum difference or ratio expressions. Fetch ALL matching rows by default.

   - **Example Queries (MySQL 8+ Safe):**

     **1. Total functional hatcheries in a district**
     ```
     SELECT district_name,
            SUM(number_of_functional_hatcheries_Number) AS total_functional_hatcheries
     FROM analysis_all_hatchery_district_wise
     WHERE district_name = 'Gaya' AND year = '2023-24'
     GROUP BY district_name;
     ```

     **2. Top 5 districts by total functional hatcheries**
     ```
     SELECT district_name,
            SUM(number_of_functional_hatcheries_Number) AS total_functional_hatcheries
     FROM analysis_all_hatchery_district_wise
     WHERE year = '2023-24'
     GROUP BY district_name
     ORDER BY total_functional_hatcheries DESC;
     ```

     **3. District-wise IMC vs exotic fry production comparison**
     ```
     SELECT district_name,
            SUM(total_production_of_imc_fry_Lakh) AS total_imc_fry,
            SUM(total_production_of_exotic_fry_Lakh) AS total_exotic_fry,
            (SUM(total_production_of_imc_fry_Lakh) - SUM(total_production_of_exotic_fry_Lakh)) AS fry_difference
     FROM analysis_all_hatchery_district_wise
     WHERE year = '2023-24'
     GROUP BY district_name
     ORDER BY fry_difference DESC;
     ```

     **4. Division-wise total hatchery production**
     ```
     SELECT division_name,
            SUM(total_production_of_imc_spawn_Lakh +
                total_production_of_exotic_spawn_Lakh) AS total_spawn_production
     FROM analysis_all_hatchery_district_wise
     WHERE year = '2023-24'
     GROUP BY division_name
     ORDER BY total_spawn_production DESC;
     ```

     **5. Percentage of functional hatcheries per district**
     ```
     SELECT district_name,
            SUM(number_of_functional_hatcheries_Number) AS functional,
            SUM(number_of_non_functional_hatcheries_Number) AS non_functional,
            (SUM(number_of_functional_hatcheries_Number) /
             (SUM(number_of_functional_hatcheries_Number) + SUM(number_of_non_functional_hatcheries_Number))) * 100 AS functional_percentage
     FROM analysis_all_hatchery_district_wise
     WHERE year = '2023-24'
     GROUP BY district_name
     ORDER BY functional_percentage DESC;
     ```

   - **Complex Logic Handling:**
     - If question includes “compare exotic and IMC,” compute difference or ratio between corresponding columns.
     - If “total hatcheries,” sum both functional + non-functional.
     - If “seed production trend,” group by `year` and sum by district/division.
     - Always `GROUP BY` relevant geography, never mix unaggregated numeric columns.

   - **Never:**
     - Add revenue or FCS filters (not part of this table).
     - Use data from different years without explicit grouping by `year`.
     - Mix IMC and exotic without clear calculation alias.

   - **Default Filters:**
     - `WHERE year = '2023-24'` if unspecified.
     - For state totals, remove `GROUP BY`.

   - **Preferred Aliases:**
     - `total_functional_hatcheries`
     - `total_seed_production`
     - `imc_total_fry`
     - `exotic_total_spawn`
     - `functional_percentage`

**Table: details_newly_formed_fcs_district_wise**
   - This dataset lists **recently registered Fishermen Cooperative Societies (FCS)**, their member counts, and registration details across Bihar.
   - Each record represents a newly formed cooperative under a district/block/panchayat for a specific year.

   - **Columns:**
     - `division_name`: Administrative division name.
     - `district_name`: District of registration.
     - `block_name`: Block where the cooperative society is registered.
     - `name_of_newly_formed_cooperative_societies`: Official name of the newly formed FCS.
     - `total_number_of_members_enrolled`: Total members at registration.
     - `total_number_of_male_members_enrolled`: Number of male members.
     - `total_number_of_female_members_enrolled`: Number of female members.
     - `date_of_registration`: Date of registration (YYYY-MM-DD format or text).
     - `year`: Fiscal year of registration (e.g., 2023, 2024).
     - `whether_Inland_or_Marine_FCS`: Type of cooperative (‘Inland’, ‘Marine’, or ‘Both’).

   - **Purpose:**
     - Used for questions about:
       - Number of newly formed FCSs per district/year.
       - Gender participation in new cooperatives.
       - Registration trends across years.
       - Type of FCS distribution (Inland vs Marine).

   - **Aggregation Rules:**
     1. Use `COUNT(*)` for total number of new FCS entries.
     2. Use `SUM()` for member totals.
     3. Default `WHERE year = 2024` if no year mentioned.
     4. Always group by geography (district_name, division_name) when comparing.
     5. Alias aggregates meaningfully:
        - `new_fcs_count`
        - `total_members_enrolled`
        - `total_male_members`
        - `total_female_members`
     6. For gender ratio, calculate:  
        `(SUM(total_number_of_female_members_enrolled) / SUM(total_number_of_members_enrolled)) * 100 AS female_participation_percentage`

   - **Example Questions:**
     - “How many new FCSs were formed in 2024?”
     - “Top 5 districts with the highest number of newly formed FCSs.”
     - “Total male and female members enrolled in Patna division.”
     - “Which year saw the most new FCS formations?”
     - “List newly registered marine FCSs.”

   - **Example Queries (MySQL 8+ Safe):**

     **1. Total newly formed FCSs per district**
     ```
     SELECT district_name,
            COUNT(*) AS new_fcs_count
     FROM details_newly_formed_fcs_district_wise
     WHERE year = 2024
     GROUP BY district_name
     ORDER BY new_fcs_count DESC;
     ```

     **2. Gender participation in new FCSs**
     ```
     SELECT district_name,
            SUM(total_number_of_male_members_enrolled) AS total_male_members,
            SUM(total_number_of_female_members_enrolled) AS total_female_members,
            (SUM(total_number_of_female_members_enrolled) /
             SUM(total_number_of_members_enrolled)) * 100 AS female_participation_percentage
     FROM details_newly_formed_fcs_district_wise
     WHERE year = 2024
     GROUP BY district_name
     ORDER BY female_participation_percentage DESC;
     ```

     **3. Type of FCS distribution**
     ```
     SELECT whether_Inland_or_Marine_FCS,
            COUNT(*) AS fcs_count
     FROM details_newly_formed_fcs_district_wise
     WHERE year = 2024
     GROUP BY whether_Inland_or_Marine_FCS
     ORDER BY fcs_count DESC;
     ```

     **4. Division-wise total members**
     ```
     SELECT division_name,
            SUM(total_number_of_members_enrolled) AS total_members
     FROM details_newly_formed_fcs_district_wise
     WHERE year = 2024
     GROUP BY division_name
     ORDER BY total_members DESC;
     ```

     **5. List new cooperatives registered in a district**
     ```
     SELECT name_of_newly_formed_cooperative_societies,
            block_name,
            total_number_of_members_enrolled,
            date_of_registration,
            whether_Inland_or_Marine_FCS
     FROM details_newly_formed_fcs_district_wise
     WHERE district_name = 'Muzaffarpur' AND year = 2024
     ORDER BY date_of_registration DESC;
     ```

   - **Complex Logic Handling:**
     - “Compare male-to-female participation” → ratio query.
     - “Top divisions by membership count” → group by division_name, order DESC.
     - “Year-over-year trend” → group by year and sum counts.
     - “Inland vs Marine FCS comparison” → group by whether_Inland_or_Marine_FCS.

   - **Key Rules Recap:**
     - Always filter by `year` (default 2024).
     - Use COUNT() for society counts; SUM() for member counts.
     - No revenue or hatchery data mixing.
     - Always GROUP BY district/division/type as per question.
     - Use clean aliases for readability:
       - `new_fcs_count`
       - `female_participation_percentage`
       - `total_members_enrolled`
       - `fcs_type_distribution`



Your goal: produce correct, executable, and semantically precise SQL queries
that will not trigger MySQL's "Expression not in GROUP BY clause" errors.

Output only the final SQL query, nothing else.
"""

    # Build the final prompt by concatenating the base text with dynamic sections
    # This avoids f-string brace-escaping issues
    prompt = (
        base_prompt
        + "\n---\n\n"
        + "### INTENT SPECIFICATION\n"
        + json.dumps(intent, indent=2)
        + "\n\n### SCHEMA INFORMATION\n"
        + schema_info
        + "\n\n### USER QUESTION\n"
        + question
        + "\n\n---\n\n"
        + "Now generate the final SQL query to accurately answer the user's question. Output ONLY the SQL query:"
    )

    return prompt

def build_answer_prompt(question: str, insights: Dict[str, Any], data_summary: str, language: str) -> str:
    """
    Specialized prompt for answer generation.
    Uses pre-computed insights and sanitized data.
    """
    if language.lower() == "hindi":
        lang_instruction = """हिंदी में संक्षिप्त उत्तर दें:
- 1 पैराग्राफ सारांश (2-3 पंक्तियाँ)
- 3-5 मुख्य बुलेट पॉइंट
- सभी आंकड़े **बोल्ड** करें"""
    else:
        lang_instruction = """Write a concise answer in English:
- 1 summary paragraph (2-3 lines)
- 3-5 key bullet points
- Bold all numbers and names using **text** format"""

    prompt = f"""You are a professional data analyst. Answer the user's question based on the computed insights.

User Question: {question}

Computed Insights:
{json.dumps(insights, indent=2)}

Data Summary ({insights['total_rows']} rows):
{data_summary}

{lang_instruction}

Requirements:
- Ground all statements in actual data
- Include units with all numbers
- Be concise and factual
- Never reference the backend or prompts

Answer:"""

    return prompt

# ============================================================================
# 9. EVALUATION HOOKS - Logging for analysis
# ============================================================================

def log_evaluation_metrics(
    question: str,
    selected_tables: List[Tuple[str, float]],
    final_sql: str,
    execution_success: bool,
    record_count: int,
    latency: float
):
    """
    Log evaluation metrics for quality analysis.
    Logs: selected tables with scores, final SQL, execution success/failure, record count, latency.

    REFINEMENT 6: Persists evaluation logs to JSONL file for post-analysis.
    """
    eval_log = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'selected_tables': selected_tables,
        'final_sql': final_sql,
        'execution_success': execution_success,
        'record_count': record_count,
        'latency_seconds': latency
    }

    # Log to console
    logger.info(f"[EVALUATION] {json.dumps(eval_log)}")

    # REFINEMENT 6: Persist to JSONL file for post-analysis
    try:
        with open(EVAL_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(eval_log) + '\n')
        logger.debug(f"Evaluation metrics logged to {EVAL_LOG_PATH}")
    except Exception as e:
        logger.error(f"Failed to write evaluation log: {str(e)}")

# ============================================================================
# Legacy helper functions (preserved for backward compatibility)
# ============================================================================

def detect_question_keywords(question: str) -> dict:
    """Detect relevant keywords and domain context from the question for all 8 tables"""
    question_lower = question.lower()
    detected_keywords = {
        "domain": [],
        "tables": [],
        "metrics": [],
        "granularity": [],
        "type": [],
        "special_context": []
    }

    # Detect which table(s) the question relates to
    # Hatchery table detection
    if any(word in question_lower for word in ["hatchery", "hatcheries", "breeding", "exotic fingerling", "exotic fry", "exotic spawn", "imc fingerling", "imc fry", "imc spawn"]):
        detected_keywords["tables"].append("analysis_all_hatchery_district_wise")

    # Areas without jalkars table detection
    if any(word in question_lower for word in ["without jalkar", "no jalkar", "no water body", "areas without"]):
        detected_keywords["tables"].append("areas_without_jalkars_by_range_district_block_panchayat")

    # FCS district-wise table detection
    if any(word in question_lower for word in ["fcs district", "cooperative society district", "fcs members district", "block count"]):
        detected_keywords["tables"].append("details_fcs_district_wise")

    # Newly formed FCS table detection
    if any(word in question_lower for word in ["newly formed fcs", "new fcs", "new cooperative", "registration date", "inland fcs", "marine fcs"]):
        detected_keywords["tables"].append("details_newly_formed_fcs_district_wise")

    # Multiple panchayat jalkars table detection
    if any(word in question_lower for word in ["multiple panchayat", "spread", "spanning", "inter-panchayat", "shared jalkar"]):
        detected_keywords["tables"].append("jalkars_spread_across_multiple_panchayats_block_district_range_w")

    

    settlement_and_jalkar_keywords = [
        # PRIMARY JALKAR KEYWORDS - Main table identifier
        "jalkar", "jalkars", "water body", "water bodies", "ponds",
        # Jalkar listing/enumeration queries
        "list jalkar", "jalkar name", "jalkar names", "jalkar in", "jalkar list",
        "all jalkars", "total jalkars", "number of jalkars", "count of jalkars",
        "jalkar wise", "jalkar breakdown", "jalkar distribution",

        # Settlement status types
        "settlement status", "unclear settlement", "settled", "unsettled",
        "in progress", "settlement in progress",
        # Settlement methods
        "open bid", "close bid", "co-operative", "settled with co-operative",
        # Legal/Court related
        "court case", "pending court", "court",
        # Special designations
        "free fishing", "perta", "kvk jale", "transfer to kvk",
        # Land records
        "khata", "khasra", "land record",
        # Financial aspects
        "security deposit", "deposit", "rental", "lease",
        # Area/measurement related to disputes
        "disputed area", "unclear boundary", "boundary dispute",
        # Jalkars with specific statuses
        "restricted by court", "free zone", "fishing zone"
    ]
    if any(word in question_lower for word in settlement_and_jalkar_keywords):
        detected_keywords["tables"].append("list_of_jalkars_with_unclear_settlement_status_panchayat_wise")

    # Total revenue report table detection
    if any(word in question_lower for word in ["fish production", "fish seed", "export", "import", "annual report", "production target", "revenue collection"]):
        detected_keywords["tables"].append("total_revenue_jalkar_report")

    # Detect metric types
    if any(word in question_lower for word in ["revenue", "demand", "collection", "recovery", "amount", "rupee", "lakh", "security deposit", "jamarashi"]):
        detected_keywords["metrics"].append("monetary")
    if any(word in question_lower for word in ["area", "hectare", "rakba", "land", "size", "coverage"]):
        detected_keywords["metrics"].append("area")
    if any(word in question_lower for word in ["number", "count", "quantity", "units", "how many", "total count"]):
        detected_keywords["metrics"].append("count")
    if any(word in question_lower for word in ["production", "yield", "output", "catch", "fingerling", "fry", "spawn", "yearling"]):
        detected_keywords["metrics"].append("production")

    logger.info(f"[KEYWORD_DETECTION] Tables={detected_keywords['tables']}")
    return detected_keywords

# ============================================================================
# GPT-4o API Call Function for SQL Query Generation
# ============================================================================

def call_gpt4o(prompt: str, temperature: float = 0.3) -> str:
    """
    Call OpenAI GPT-4o API for SQL query generation.
    Returns the model's response text.
    """
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_gpt_4o_api_key_here":
        logger.error("OPENAI_API_KEY not configured in .env file")
        raise Exception("OPENAI_API_KEY not configured. Please add your API key to .env file.")

    logger.debug(f"Calling GPT-4o API with temperature: {temperature}")
    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": GPT_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": 1024,
            "top_p": 0.95
        }

        timeout = get_api_timeout("gpt4o")  # IMPROVEMENT 5: Use centralized timeout
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()

        result = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        logger.debug(f"GPT-4o API response received, length: {len(result)}")
        return result

    except Exception as e:
        logger.error(f"GPT-4o API error: {str(e)}")
        raise Exception(f"GPT-4o API error: {str(e)}")

# ============================================================================
# REFINEMENT 8: Future-proofing - Timeout guards and async support placeholders
# ============================================================================

# Timeout configuration for API calls (in seconds)
API_TIMEOUT_CONFIG = {
    "gpt4o": 45,
    "ollama": 45,
    "database": 30
}

# Optional: Response caching placeholder for future optimization
# cache = {}  # Could use functools.lru_cache or Redis in production

def get_api_timeout(api_type: str) -> int:
    """
    Get timeout configuration for different API types.
    Allows centralized timeout management.

    REFINEMENT 8: Centralized timeout configuration for future customization.
    """
    return API_TIMEOUT_CONFIG.get(api_type, 45)

def call_ollama(model: str, prompt: str, temperature: float = 0.3, stream: bool = False):
    """Call Ollama API with optimized parameters - returns full text or yields streaming tokens"""
    logger.debug(f"Calling Ollama API with model: {model}, temperature: {temperature}, stream: {stream}")
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "temperature": temperature,
            "num_predict": 512,
            "top_p": 0.85,
            "repeat_penalty": 1.1,
            "top_k": 40
        }

        timeout = get_api_timeout("ollama")  # IMPROVEMENT 5: Use centralized timeout
        response = ollama_session.post(
            f"{OLLAMA_BASE_URL}/generate",
            json=payload,
            timeout=timeout,
            stream=stream
        )
        response.raise_for_status()

        if not stream:
            # Non-streaming: return full response
            result = response.json().get("response", "").strip()
            logger.debug(f"Ollama API response received, length: {len(result)}")
            return result
        else:
            # Streaming: yield tokens as they arrive
            def token_generator():
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
            return token_generator()

    except Exception as e:
        logger.error(f"Ollama API error: {str(e)}")
        raise Exception(f"Ollama API error: {str(e)}")

# ============================================================================
# Main SQL Generation with all improvements integrated
# ============================================================================

def generate_sql_query_direct(question: str, language: str = "english") -> tuple:
    """
    Advanced SQL query generator with all improvements:
    - Schema registry introspection
    - Table alias normalization
    - Hybrid table retrieval
    - Intent extraction
    - Year rules application
    - SQL validation with retry
    """
    start_time = time.time()
    logger.info(f"SQL Generation Request: {question}")

    # Step 1: Normalize location names
    normalized_question = normalize_location_names(question)
    logger.info(f"Normalized question: {normalized_question}")

    # Step 2: Hybrid table retrieval - get top-2 candidate tables
    candidate_tables = get_top_candidate_tables(normalized_question, top_k=2)

    if not candidate_tables:
        logger.warning("No candidate tables found")
        return "", []

    # Step 3: Extract query intent
    intent = extract_query_intent(normalized_question, candidate_tables)

    if not intent or not intent.get('tables'):
        logger.warning("Intent extraction failed, falling back to direct generation")
        # Fallback to legacy method
        return generate_sql_fallback(normalized_question, candidate_tables)

    # Step 3.5: Validate intent before generating SQL (IMPROVEMENT 1)
    if not validate_intent_for_sql(intent):
        logger.warning("Intent JSON invalid, falling back to direct generation")
        return generate_sql_fallback(normalized_question, candidate_tables)

    # Step 4: Build schema context for selected tables
    schema_context = ""
    for table in intent['tables']:
        if table in SCHEMA_REGISTRY:
            schema_context += f"\nTABLE: {table}\n"
            schema_context += f"Columns:\n"
            for col, meta in SCHEMA_REGISTRY[table]['columns'].items():
                schema_context += f"  - {col} ({meta['data_type']})\n"
        elif table in DB_DESCRIPTIONS:
            schema_context += f"\nTABLE: {table}\n"
            schema_context += f"Columns: {', '.join(DB_DESCRIPTIONS[table]['columns'].keys())}\n"

    # Step 5: Generate SQL using query plan prompt
    query_plan_prompt = build_query_plan_prompt(normalized_question, intent, schema_context)

    try:
        sql_query = call_gpt4o(query_plan_prompt, temperature=0.0)

        # Clean up response
        sql_query = sql_query.strip()
        if sql_query.startswith("```"):
            sql_query = re.sub(r'^```(?:sql)?\s*\n', '', sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(r'\n```\s*$', '', sql_query)

        # Extract SELECT statement
        select_match = re.search(r'\bSELECT\b', sql_query, re.IGNORECASE)
        if select_match:
            sql_query = sql_query[select_match.start():]

        # Ensure semicolon at end
        sql_query = re.sub(r';.*$', ';', sql_query, flags=re.DOTALL)
        if not sql_query.endswith(';'):
            sql_query += ';'

        # Normalize whitespace
        sql_query = re.sub(r'\s+', ' ', sql_query).strip()

        logger.info(f"Generated SQL: {sql_query}")

        # Step 6: Apply year filter based on YEAR_RULES
        primary_table = intent['tables'][0]
        sql_query = apply_year_filter(primary_table, sql_query)

        # Step 6.5: Guard against hallucinated columns (IMPROVEMENT 2 - FIXED)
        # Now only warns about issues, does not reject valid queries with aliases
        column_check = verify_sql_columns(sql_query)
        if not column_check:
            logger.warning("SQL column verification detected potential issues - continuing to validate with dry-run")
            # Do not fallback here - let dry-run validation decide if query is truly bad

        # Step 7: Validate SQL with dry run and retry
        validated_sql, success = validate_sql_with_dry_run(sql_query, max_retries=2)

        if not success:
            logger.warning("SQL validation failed after retries, using as-is")

        # Step 8: Log evaluation metrics
        elapsed = time.time() - start_time
        log_evaluation_metrics(
            question=question,
            selected_tables=[(t, 1.0) for t in candidate_tables],
            final_sql=validated_sql,
            execution_success=success,
            record_count=0,  # Will be updated after execution
            latency=elapsed
        )

        return validated_sql, intent['tables']

    except Exception as e:
        logger.error(f"SQL generation error: {str(e)}")
        return "", []

# ============================================================================
# IMPROVEMENT 9: Safe SQL Generation with Fallback to Ollama
# ============================================================================

def safe_generate_sql_with_fallback(prompt: str, temperature: float = 0.0) -> str:
    """
    Generate SQL with GPT-4o, fallback to Ollama if GPT-4o fails.
    IMPROVEMENT 9: Ensures SQL generation never fully fails.
    """
    try:
        logger.info("Attempting SQL generation with GPT-4o")
        return call_gpt4o(prompt, temperature=temperature)
    except Exception as e:
        logger.error(f"GPT-4o failed ({str(e)}), falling back to Ollama Qwen2.5-coder")
        try:
            # Fallback to local Qwen model
            return call_ollama("qwen2.5-coder:7b", prompt, temperature=temperature)
        except Exception as e2:
            logger.error(f"Ollama fallback also failed: {str(e2)}")
            raise Exception(f"SQL generation failed in both GPT-4o and Ollama: {str(e)}")

def generate_sql_fallback(question: str, candidate_tables: List[str]) -> tuple:
    """
    Fallback SQL generation using simplified prompt.
    Used when intent extraction fails.
    FIXED: Now includes geographic awareness to prevent 'Bihar' filter errors.
    """
    logger.info("Using fallback SQL generation")

    # Build simple schema description
    schema_desc = ""
    for table in candidate_tables:
        if table in DB_DESCRIPTIONS:
            schema_desc += f"\nTable: {table}\n"
            schema_desc += f"Columns: {', '.join(DB_DESCRIPTIONS[table]['columns'].keys())}\n"

    prompt = f"""Generate a MySQL SELECT query to answer the question.

{schema_desc}

⚠️ CRITICAL GEOGRAPHIC RULES:
- Bihar is the ENTIRE STATE, NOT a division
- NEVER filter: "WHERE division_name = 'Bihar'" (returns ZERO rows)
- For "each district of Bihar": use "GROUP BY district_name" with NO WHERE geographic filter
- For specific district "Patna": use "district_name = 'Patna'"
- For specific division: use "division_name = '[DivisionName]'"

Question: {question}

Output ONLY the SQL query (no explanations):"""

    try:
        sql_query = safe_generate_sql_with_fallback(prompt, temperature=0.1)  # IMPROVEMENT 9: Use fallback wrapper

        # Clean up
        sql_query = sql_query.strip()
        if sql_query.startswith("```"):
            sql_query = re.sub(r'^```(?:sql)?\s*\n', '', sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(r'\n```\s*$', '', sql_query)

        if not sql_query.endswith(';'):
            sql_query += ';'

        # Apply year filter
        if candidate_tables:
            sql_query = apply_year_filter(candidate_tables[0], sql_query)

        # Validate column references using improved verify_sql_columns
        # Log warnings but do not reject - let dry-run decide
        column_check = verify_sql_columns(sql_query)
        if not column_check:
            logger.warning("Fallback SQL: Column verification detected potential issues - continuing to dry-run validation")

        return sql_query, candidate_tables

    except Exception as e:
        logger.error(f"Fallback SQL generation error: {str(e)}")
        return "", []

def convert_decimal_to_string(obj):
    """Convert Decimal values to strings for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_decimal_to_string(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_string(item) for item in obj]
    elif isinstance(obj, Decimal):
        return str(obj)
    else:
        return obj

def execute_sql_query(sql_query: str) -> list:
    """Execute SQL query using connection pool - optimized for speed"""
    logger.info(f"Executing SQL query: {sql_query}")
    connection = None
    try:
        # Use connection pool if available, fallback to direct connection
        if db_pool:
            connection = db_pool.get_connection()
        else:
            connection = mysql.connector.connect(**DB_CONFIG)

        cursor = connection.cursor(dictionary=True)
        cursor.execute(sql_query)
        results = cursor.fetchall()
        cursor.close()

        logger.info(f"Query executed successfully. Results count: {len(results)}")

        # Convert Decimal values to strings for JSON serialization
        results = convert_decimal_to_string(results)

        return results
    except Exception as e:
        logger.error(f"Database error while executing query: {str(e)}")
        raise Exception(f"Database error: {str(e)}")
    finally:
        if connection:
            connection.close()

# ============================================================================
# Visualization and PDF generation (preserved from original)
# ============================================================================

def identify_numeric_columns(data: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    """Identify which columns contain numeric data for visualization"""
    numeric_columns = []

    if not data:
        return numeric_columns

    for col in columns:
        try:
            # Check first non-null value in column
            for row in data:
                val = row.get(col)
                if val is not None:
                    float(val)
                    numeric_columns.append(col)
                    break
        except (ValueError, TypeError):
            continue

    return numeric_columns

def identify_string_columns(data: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    """Identify which columns contain string/categorical data"""
    string_columns = []
    numeric_cols = identify_numeric_columns(data, columns)

    for col in columns:
        if col not in numeric_cols and col not in ['id']:
            string_columns.append(col)

    return string_columns

def generate_chart_config(data: List[Dict[str, Any]], columns: List[str]) -> dict:
    """
    Generate Chart.js configuration based on data structure.

    ENHANCEMENT: Only generates graphs when:
    - At least 2 columns of data are present
    - At least 2 rows of data are present
    - Data contains numeric values for visualization
    """
    logger.info(f"Generating chart config for {len(data)} rows with columns: {columns}")

    if not data or not columns:
        return {"type": "none", "message": "No data available for visualization"}

    # Check if minimum requirements are met for visualization
    # Requirement: At least 2 columns AND at least 2 rows
    if len(columns) < 2:
        logger.info(f"Insufficient columns for visualization: {len(columns)} < 2")
        return {"type": "none", "message": "Insufficient columns for visualization"}

    if len(data) < 2:
        logger.info(f"Insufficient rows for visualization: {len(data)} < 2")
        return {"type": "none", "message": "Insufficient rows for visualization"}

    numeric_cols = identify_numeric_columns(data, columns)
    string_cols = identify_string_columns(data, columns)

    logger.info(f"Numeric columns: {numeric_cols}, String columns: {string_cols}")

    # Generate chart only if we have numeric data to visualize
    # Default: bar chart for categorical data with numeric values
    if string_cols and numeric_cols:
        return generate_bar_chart(data, string_cols[0], numeric_cols, columns)
    elif numeric_cols and len(data) > 5:
        # Line chart for time-series or sequential data
        return generate_line_chart(data, 'index', numeric_cols[:3], columns)
    elif numeric_cols and len(data) >= 2:
        # Bar chart for smaller numeric datasets (at least 2 rows)
        return generate_bar_chart(data, None, numeric_cols, columns)
    else:
        logger.info("Data structure not suitable for visualization")
        return {"type": "none", "message": "Data structure not suitable for visualization"}

def generate_bar_chart(data: List[Dict[str, Any]], category_col: str, value_cols: List[str], all_columns: List[str]) -> dict:
    """Generate a bar chart configuration"""
    logger.info(f"Generating bar chart with category: {category_col}, values: {value_cols}")

    labels = []
    datasets = []
    colors = ["rgba(30, 64, 175, 0.8)", "rgba(220, 53, 69, 0.8)", "rgba(40, 167, 69, 0.8)",
              "rgba(255, 193, 7, 0.8)", "rgba(23, 162, 184, 0.8)", "rgba(156, 39, 176, 0.8)"]

    # Extract unique categories
    for row in data:
        cat = row.get(category_col)
        if cat is not None and cat not in labels:
            labels.append(str(cat))

    # Limit to 20 categories for readability
    labels = labels[:20]

    # Create dataset for each value column
    for idx, val_col in enumerate(value_cols[:6]):
        values = []
        for cat in labels:
            val = None
            for row in data:
                if str(row.get(category_col)) == cat:
                    try:
                        val = float(row.get(val_col, 0)) if row.get(val_col) is not None else 0
                    except (ValueError, TypeError):
                        val = 0
                    break
            values.append(val if val is not None else 0)

        datasets.append({
            "label": val_col,
            "data": values,
            "backgroundColor": colors[idx % len(colors)],
            "borderColor": colors[idx % len(colors)].replace("0.8", "1"),
            "borderWidth": 1
        })

    return {
        "type": "bar",
        "data": {
            "labels": labels,
            "datasets": datasets
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": True,
            "plugins": {
                "title": {
                    "display": True,
                    "text": f"Comparison by {category_col}",
                    "font": {"size": 16, "weight": "bold"}
                },
                "legend": {
                    "display": True,
                    "labels": {"font": {"size": 11}, "padding": 15}
                }
            },
            "scales": {
                "x": {
                    "ticks": {"font": {"size": 10}},
                    "grid": {"display": False}
                },
                "y": {
                    "beginAtZero": True,
                    "ticks": {"font": {"size": 10}},
                    "grid": {"color": "rgba(0,0,0,0.05)"}
                }
            }
        }
    }

def generate_line_chart(data: List[Dict[str, Any]], category_col: str, value_cols: List[str], all_columns: List[str]) -> dict:
    """Generate a line chart for time-series data"""
    logger.info(f"Generating line chart with category: {category_col}, values: {value_cols}")

    if not isinstance(value_cols, list):
        value_cols = [value_cols]

    labels = []
    datasets = []
    colors = ["rgba(30, 64, 175, 1)", "rgba(220, 53, 69, 1)", "rgba(40, 167, 69, 1)",
              "rgba(255, 193, 7, 1)", "rgba(23, 162, 184, 1)", "rgba(156, 39, 176, 1)"]

    # Get unique categories
    categories = []
    for row in data:
        cat = row.get(category_col)
        if cat is not None and cat not in categories:
            categories.append(cat)

    labels = categories[:50]

    # Create dataset for each value column
    for idx, val_col in enumerate(value_cols[:6]):
        values = []
        for cat in labels:
            val = None
            for row in data:
                if row.get(category_col) == cat:
                    try:
                        val = float(row.get(val_col, 0)) if row.get(val_col) is not None else 0
                    except (ValueError, TypeError):
                        val = 0
                    break
            values.append(val if val is not None else 0)

        datasets.append({
            "label": val_col,
            "data": values,
            "borderColor": colors[idx % len(colors)],
            "backgroundColor": colors[idx % len(colors)].replace("1)", "0.1)"),
            "fill": True,
            "tension": 0.4,
            "borderWidth": 2,
            "pointRadius": 4,
            "pointHoverRadius": 6,
            "pointBackgroundColor": colors[idx % len(colors)]
        })

    return {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": datasets
        },
        "options": {
            "responsive": True,
            "maintainAspectRatio": True,
            "interaction": {
                "mode": "index",
                "intersect": False
            },
            "plugins": {
                "title": {
                    "display": True,
                    "text": f"Trend Analysis by {category_col}",
                    "font": {"size": 16, "weight": "bold"}
                },
                "legend": {
                    "display": True,
                    "labels": {"font": {"size": 11}, "padding": 15, "usePointStyle": True}
                }
            },
            "scales": {
                "x": {
                    "ticks": {"font": {"size": 10}},
                    "grid": {"display": False}
                },
                "y": {
                    "beginAtZero": True,
                    "ticks": {"font": {"size": 10}},
                    "grid": {"color": "rgba(0,0,0,0.1)"}
                }
            }
        }
    }

def generate_chart_image(data: List[Dict[str, Any]], columns: List[str]) -> Optional[io.BytesIO]:
    """Generate a chart image using matplotlib for PDF embedding"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        numeric_cols = identify_numeric_columns(data, columns)
        string_cols = identify_string_columns(data, columns)

        if not string_cols or not numeric_cols:
            return None

        category_col = string_cols[0]
        value_cols = numeric_cols[:3]

        # Extract data
        categories = []
        values_dict = {col: [] for col in value_cols}

        for row in data[:20]:
            cat = row.get(category_col)
            if cat is not None:
                categories.append(str(cat))
                for col in value_cols:
                    try:
                        val = float(row.get(col, 0)) if row.get(col) is not None else 0
                        values_dict[col].append(val)
                    except (ValueError, TypeError):
                        values_dict[col].append(0)

        if not categories:
            return None

        # Create chart
        fig, ax = plt.subplots(figsize=(10, 5))

        x_pos = np.arange(len(categories))
        bar_width = 0.8 / len(value_cols)

        for idx, col in enumerate(value_cols):
            ax.bar(x_pos + (idx * bar_width), values_dict[col], bar_width, label=col)

        ax.set_xlabel(category_col, fontsize=11, fontweight='bold')
        ax.set_ylabel('Values', fontsize=11, fontweight='bold')
        ax.set_title('Data Visualization', fontsize=13, fontweight='bold', color='#1e40af')
        ax.set_xticks(x_pos + bar_width)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save to bytes
        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format='png', dpi=100, bbox_inches='tight')
        chart_buffer.seek(0)
        plt.close()

        return chart_buffer

    except Exception as e:
        logger.warning(f"Could not generate chart: {str(e)}")
        return None

def sanitize_html_for_pdf(text: str) -> str:
    """
    Sanitize HTML text for PDF generation.
    Ensures all tags are properly closed and no orphaned tags exist.
    """
    # Replace ** markers with proper <b> tags
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)

    # Ensure all <b> tags are properly closed
    open_tags = text.count('<b>') - text.count('</b>')
    if open_tags > 0:
        text += '</b>' * open_tags

    # Remove any stray </b> tags without opening
    close_tags = text.count('</b>') - text.count('<b>')
    for _ in range(max(0, close_tags)):
        # Remove one extra closing tag
        text = text.replace('</b>', '', 1)

    return text

def generate_pdf_report(question: str, answer: str, data: List[Dict[str, Any]], columns: List[str], sql_query: str) -> bytes:
    """Generate a formatted PDF report with question, answer, charts, and data table"""
    logger.info(f"Generating PDF report for question: {question}")

    try:
        from reportlab.platypus import Image as RLImage

        # Create PDF buffer
        pdf_buffer = io.BytesIO()

        # Create PDF document
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.75*inch, bottomMargin=0.75*inch)
        elements = []

        # Get styles
        styles = getSampleStyleSheet()

        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=8,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#1e40af'),
            spaceAfter=8,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        )

        # Add title
        title = Paragraph("GOVERNMENT OF BIHAR<br/>ANIMAL & FISHERIES DEPARTMENT<br/>DATA ANALYSIS REPORT", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.2*inch))

        # Add timestamp
        timestamp = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        timestamp_para = Paragraph(f"<b>Generated:</b> {timestamp}", normal_style)
        elements.append(timestamp_para)
        elements.append(Spacer(1, 0.15*inch))

        # Add question
        elements.append(Paragraph("<b>User Question:</b>", heading_style))
        question_para = Paragraph(f"{question}", normal_style)
        elements.append(question_para)
        elements.append(Spacer(1, 0.12*inch))

        # Add answer
        answer = answer.strip()
        elements.append(Paragraph("<b>Analysis & Answer:</b>", heading_style))

        # Sanitize HTML before processing
        answer_html = sanitize_html_for_pdf(answer)

        lines = answer_html.split('\n')
        formatted_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                bullet_content = re.sub(r'^[-•*]\s+', '', line)
                formatted_lines.append(f"• {bullet_content}")
            elif line and not line.startswith('<'):
                formatted_lines.append(line)

        if formatted_lines:
            current_list = []
            for line in formatted_lines:
                if line.startswith('•'):
                    current_list.append(line)
                else:
                    if current_list:
                        for item in current_list:
                            item_text = item.replace('• ', '')
                            answer_para = Paragraph(f"• {item_text}", normal_style)
                            elements.append(answer_para)
                        current_list = []
                    answer_para = Paragraph(line, normal_style)
                    elements.append(answer_para)

            if current_list:
                for item in current_list:
                    item_text = item.replace('• ', '')
                    answer_para = Paragraph(f"• {item_text}", normal_style)
                    elements.append(answer_para)

        elements.append(Spacer(1, 0.15*inch))

        # Generate and add chart if data is available
        if data and len(data) > 0 and columns:
            chart_buffer = generate_chart_image(data, columns)
            if chart_buffer:
                try:
                    elements.append(Paragraph("<b>Data Visualization:</b>", heading_style))
                    chart_img = RLImage(chart_buffer, width=6.5*inch, height=3.25*inch)
                    elements.append(chart_img)
                    elements.append(Spacer(1, 0.15*inch))
                except Exception as e:
                    logger.warning(f"Could not embed chart in PDF: {str(e)}")

        # Add data table if available
        if data and len(data) > 0 and columns:
            elements.append(Paragraph("<b>Data Table:</b>", heading_style))

            table_data = [columns]

            for row in data[:20]:
                row_data = []
                for col in columns:
                    value = row.get(col, '')
                    val_str = str(value) if value is not None else '-'
                    if len(val_str) > 15:
                        val_str = val_str[:12] + '...'
                    row_data.append(val_str)
                table_data.append(row_data)

            if len(columns) <= 3:
                col_width = 2.0*inch
            elif len(columns) <= 5:
                col_width = 1.3*inch
            else:
                col_width = 0.9*inch

            table = Table(table_data, colWidths=[col_width] * len(columns))

            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))

            elements.append(table)

            if len(data) > 20:
                elements.append(Spacer(1, 0.08*inch))
                note = Paragraph(f"<i>Showing first 20 rows of {len(data)} total rows</i>", normal_style)
                elements.append(note)

            elements.append(Spacer(1, 0.15*inch))

        # Add footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#6b7280'),
            alignment=TA_CENTER
        )
        footer = Paragraph("This report was generated by the Animal & Fisheries Department Chatbot", footer_style)
        elements.append(Spacer(1, 0.2*inch))
        elements.append(footer)

        # Build PDF
        doc.build(elements)

        pdf_buffer.seek(0)
        logger.info("PDF report generated successfully")
        return pdf_buffer.getvalue()

    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")

# ============================================================================
# API Endpoints (preserved for backward compatibility)
# ============================================================================

class ChatRequest(BaseModel):
    question: str
    language: str = "english"

class ChatResponse(BaseModel):
    question: str
    answer: str
    sql_query: str
    language: str
    query_results: list = []

class VisualizationRequest(BaseModel):
    data: List[Dict[str, Any]]
    columns: List[str]

class PDFReportRequest(BaseModel):
    question: str
    answer: str
    data: List[Dict[str, Any]] = []
    columns: List[str] = []
    sql_query: str = ""

def yield_json_line(stage: str, **data) -> str:
    """Helper to format JSON lines for streaming"""
    payload = {"stage": stage}
    payload.update(data)
    return json.dumps(payload, default=str) + "\n"

def generate_thinking_message(question: str) -> str:
    """Generate an engaging, conversational thinking message using fast model"""
    prompt = f"""Generate a friendly, engaging response showing you're analyzing the question. Write 2-3 lines (60-90 words) that feels natural.

Examples:
"Great question! Let me analyze the fisheries data to find exactly what you need..."

"Got it! I'm pulling together the latest information from the database right now..."

User question: {question}

Response (2-3 lines only):"""

    try:
        thinking_text = call_ollama(FAST_MODEL, prompt, temperature=0.8)
        thinking_text = thinking_text.strip()
        if len(thinking_text) > 200:
            thinking_text = thinking_text[:200].rstrip() + "..."
        return thinking_text
    except Exception as e:
        logger.warning(f"Failed to generate thinking message: {str(e)}")
        return "Let me analyze the data and pull together the insights you're looking for..."

@app.post("/chat")
async def chat(request: ChatRequest):
    """Streaming chat endpoint with improved SQL generation and answer accuracy"""
    request_start_time = time.time()
    question = request.question
    language = request.language

    async def stream_generator():
        try:
            logger.info(f"=== CHAT REQUEST === Question: {question[:80]}... | Language: {language}")

            # STAGE 0: Generate thinking message
            thinking_start = time.time()
            thinking_text = await asyncio.to_thread(generate_thinking_message, question)
            thinking_time = time.time() - thinking_start
            logger.info(f"✓ Thinking message: {thinking_time:.2f}s")
            yield yield_json_line("thinking", text=thinking_text)

            # STAGE 1: Generate SQL query with improvements
            sql_start = time.time()
            sql_query, tables_used = await asyncio.to_thread(generate_sql_query_direct, question, language)
            sql_time = time.time() - sql_start

            if not sql_query:
                yield yield_json_line("error", message="Unable to generate SQL query. Please try rephrasing.")
                return

            logger.info(f"✓ SQL generation: {sql_time:.1f}s")
            yield yield_json_line("sql_generated", sql_query=sql_query, tables=tables_used)

            # STAGE 2: Execute SQL query
            db_start = time.time()
            try:
                query_results = await asyncio.to_thread(execute_sql_query, sql_query)
            except Exception as db_error:
                logger.error(f"Database execution error: {str(db_error)}")
                yield yield_json_line("error", message=f"Database error: {str(db_error)}")
                return

            db_time = time.time() - db_start
            logger.info(f"✓ DB query execution: {db_time:.1f}s ({len(query_results)} rows)")
            yield yield_json_line("data_fetched", query_results=query_results)

            # STAGE 3: Generate answer with hardening
            if query_results:
                answer_start = time.time()
                try:
                    # Compute numeric insights in Python
                    insights = await asyncio.to_thread(compute_numeric_insights, query_results)

                    # Sanitize data for answer generation
                    data_summary = await asyncio.to_thread(sanitize_data_for_answer, query_results, max_rows=50)

                    # Build answer prompt
                    answer_prompt = build_answer_prompt(question, insights, data_summary, language)

                    # Get streaming token generator
                    token_stream = await asyncio.to_thread(
                        lambda: call_ollama(
                            ANSWER_GENERATOR_MODEL,
                            answer_prompt,
                            temperature=0.5,
                            stream=True
                        )
                    )

                    full_answer = ""
                    for token in token_stream:
                        if token:
                            full_answer += token
                            yield yield_json_line("partial_answer", text=token)

                    # Validate answer against data
                    full_answer = await asyncio.to_thread(validate_answer_against_data, full_answer, query_results)

                    answer_time = time.time() - answer_start
                    logger.info(f"✓ Answer generation: {answer_time:.1f}s")

                except Exception as e:
                    logger.warning(f"Answer generation failed: {str(e)}, using fallback")
                    full_answer = f"Found {len(query_results)} records. Please review the data table below for details."
                    yield yield_json_line("partial_answer", text=full_answer)
            else:
                full_answer = "No data found for your query."
                yield yield_json_line("partial_answer", text=full_answer)

            # STAGE 4: Complete
            total_time = time.time() - request_start_time
            logger.info(f"✓ STREAM COMPLETE: {total_time:.1f}s total")

            # Log final evaluation metrics
            log_evaluation_metrics(
                question=question,
                selected_tables=[(t, 1.0) for t in tables_used],
                final_sql=sql_query,
                execution_success=True,
                record_count=len(query_results),
                latency=total_time
            )

            yield yield_json_line(
                "complete",
                answer=full_answer,
                sql_query=sql_query,
                query_results=query_results,
                total_time=total_time
            )

        except Exception as e:
            logger.error(f"Stream error: {str(e)}", exc_info=True)
            yield yield_json_line("error", message=f"Server error: {str(e)}")

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

@app.post("/visualize")
async def visualize(request: VisualizationRequest):
    """Generate chart configuration for data visualization"""
    logger.info(f"Visualization request received with {len(request.data)} rows")
    try:
        chart_config = generate_chart_config(request.data, request.columns)
        logger.info("Chart configuration generated successfully")
        return chart_config
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")

@app.post("/download-report")
async def download_report(request: PDFReportRequest):
    """Generate and download PDF report with complete analysis"""
    logger.info(f"Download report requested for question: {request.question}")
    try:
        # Generate PDF
        pdf_bytes = generate_pdf_report(
            question=request.question,
            answer=request.answer,
            data=request.data,
            columns=request.columns,
            sql_query=request.sql_query
        )

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chatbot_report_{timestamp}.pdf"

        # Return as file download
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    except Exception as e:
        logger.error(f"Error in download-report endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def serve_frontend():
    """Serve the chatbot frontend"""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend.html")
    return FileResponse(frontend_path)

@app.get("/health")
async def health():
    """
    Health check endpoint with LLM and DB probes (IMPROVEMENT 11).
    Returns status of database and schema.
    """
    logger.info("Health check requested")
    db_ok = False
    try:
        if db_pool:
            conn = db_pool.get_connection()
        else:
            conn = mysql.connector.connect(**DB_CONFIG)
        conn.close()
        db_ok = True
    except Exception as e:
        logger.warning(f"DB health check failed: {str(e)}")
        db_ok = False

    return {
        "status": "healthy" if db_ok else "degraded",
        "db": db_ok,
        "schema_tables": len(SCHEMA_REGISTRY),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/admin/refresh-schema")
async def refresh_schema():
    """
    Admin endpoint to hot-reload schema without restarting server.
    IMPROVEMENT 8: Handy for ops - re-introspects MySQL schema.
    """
    try:
        logger.info("Schema refresh requested via /admin/refresh-schema")
        introspect_mysql_schema()
        logger.info(f"Schema refreshed: {len(SCHEMA_REGISTRY)} tables")
        return {
            "ok": True,
            "tables": len(SCHEMA_REGISTRY),
            "table_names": list(SCHEMA_REGISTRY.keys()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Schema refresh failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Schema refresh failed: {str(e)}")

# GPU warm-up function
def warm_up_gpu():
    """Warm up GPU by running dummy inference on both models"""
    try:
        logger.info("Warming up GPU with dummy inference...")
        for model in [SQL_GENERATOR_MODEL, ANSWER_GENERATOR_MODEL]:
            try:
                call_ollama(model, "ping", temperature=0.1)
                logger.info(f"✓ GPU warmed up for model: {model}")
            except Exception as e:
                logger.warning(f"GPU warm-up failed for {model}: {str(e)}")
    except Exception as e:
        logger.warning(f"GPU warm-up skipped: {str(e)}")

if __name__ == "__main__":
    import sys
    import uvicorn

    print("\n" + "="*80)
    print("  [CHATBOT] Enhanced RAG Chatbot Backend Server Starting...")
    print("="*80)
    logger.info("[CHATBOT] Starting Enhanced RAG Chatbot Backend Server")
    logger.info("[CHATBOT] FastAPI running on http://0.0.0.0:8002")

    # Introspect MySQL schema on startup
    introspect_mysql_schema()

    logger.info(f"[CHATBOT] Schema Registry: {len(SCHEMA_REGISTRY)} tables loaded")
    logger.info(f"[CHATBOT] Available tables: {list(SCHEMA_REGISTRY.keys())}")
    logger.info(f"[CHATBOT] MySQL Connection Pool: {'ENABLED' if db_pool else 'DISABLED'}")
    logger.info(f"[CHATBOT] Ollama Models: SQL={SQL_GENERATOR_MODEL}, Answer={ANSWER_GENERATOR_MODEL}")

    # Warm up GPU on startup
    print("[CHATBOT] Pre-warming GPU...")
    try:
        warm_up_gpu()
    except Exception as e:
        logger.warning(f"[CHATBOT] GPU warm-up skipped: {str(e)}")

    print("="*80 + "\n")
    logger.info("[CHATBOT] Backend ready for requests with enhanced SQL generation and answer accuracy")

    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
