
USE ROLE DNA_INF_DATA_SCIENCE;
USE WAREHOUSE DNA_DEFAULT_WH;

----------------------------------------------------------------------
-- STEP 1: CHUNK DOCUMENTS FOR CORTEX SEARCH
----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS DEV_SANDBOX.ABRETHO__RAW_FILES.CHUNKED_DOCUMENTS (
    FILE_PATH        VARCHAR,
    FILE_NAME        VARCHAR,
    CHUNK_INDEX      INT,
    CHUNK_TEXT       VARCHAR,
    CHUNKED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.CHUNKED_DOCUMENTS
    (FILE_PATH, FILE_NAME, CHUNK_INDEX, CHUNK_TEXT)
SELECT
    d.FILE_PATH,
    d.FILE_NAME,
    c.INDEX AS CHUNK_INDEX,
    c.VALUE::VARCHAR AS CHUNK_TEXT
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.FULL_DOCUMENTS d,
    LATERAL FLATTEN(input => SNOWFLAKE.CORTEX.SPLIT_TEXT_RECURSIVE_CHARACTER(
        d.CONTENT,
        'markdown', --format output
        1500, -- chunk size
        300 --overlap
    )) c
WHERE d.FILE_PATH NOT IN (
    SELECT DISTINCT FILE_PATH FROM DEV_SANDBOX.ABRETHO__RAW_FILES.CHUNKED_DOCUMENTS
);

-- count chunks per doc, chekc it makes sense (big doc >> lots of chunks)
SELECT FILE_NAME, COUNT(*) AS CHUNK_COUNT, AVG(LENGTH(CHUNK_TEXT)) AS AVG_CHUNK_LENGTH
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.CHUNKED_DOCUMENTS
GROUP BY FILE_NAME
ORDER BY FILE_NAME;

-- Visualize all chunks for a specific document
SELECT CHUNK_INDEX, LENGTH(CHUNK_TEXT) AS CHUNK_LENGTH, CHUNK_TEXT
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.CHUNKED_DOCUMENTS
WHERE FILE_NAME = '10471-INF-2025-Sustainability-Report_v11.pdf'
ORDER BY CHUNK_INDEX;

----------------------------------------------------------------------
-- STEP 2: CREATE CORTEX SEARCH SERVICES
----------------------------------------------------------------------

-- search service on chunked full document
CREATE OR REPLACE CORTEX SEARCH SERVICE DEV_SANDBOX.ABRETHO__RAW_FILES.FULL_DOCUMENT_SEARCH_SERVICE
    ON CHUNK_TEXT
    ATTRIBUTES FILE_NAME
    WAREHOUSE = DNA_DEFAULT_WH
    TARGET_LAG = '1 day'
    EMBEDDING_MODEL = 'snowflake-arctic-embed-l-v2.0'
AS (
    SELECT
        CHUNK_TEXT,
        FILE_PATH,
        FILE_NAME,
        CHUNK_INDEX
    FROM DEV_SANDBOX.ABRETHO__RAW_FILES.CHUNKED_DOCUMENTS
);

-- search service on document summaries (for high-level document discovery)
-- TODO: explore AI_AGG and AI_SUMMARIZE_AGG
CREATE OR REPLACE CORTEX SEARCH SERVICE DEV_SANDBOX.ABRETHO__RAW_FILES.SUMMARY_DOCUMENT_SEARCH_SERVICE
    ON SUMMARY
    ATTRIBUTES FILE_NAME
    WAREHOUSE = DNA_DEFAULT_WH
    TARGET_LAG = '1 day'
    EMBEDDING_MODEL = 'snowflake-arctic-embed-l-v2.0'
AS (
    SELECT
        SUMMARY,
        FILE_PATH,
        FILE_NAME
    FROM DEV_SANDBOX.ABRETHO__RAW_FILES.SUMMARY_DOCUMENTS
);

----------------------------------------------------------------------
-- VERIFY: Preview search results
----------------------------------------------------------------------

SELECT PARSE_JSON(
    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        'DEV_SANDBOX.ABRETHO__RAW_FILES.FULL_DOCUMENT_SEARCH_SERVICE',
        '{
            "query": "sustainability in 2024 compared to 2025",
            "columns": ["CHUNK_TEXT", "FILE_NAME", "CHUNK_INDEX"],
            "limit": 15
        }'
    )
)['results'] AS RESULTS;

SELECT PARSE_JSON(
    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        'DEV_SANDBOX.ABRETHO__RAW_FILES.SUMMARY_DOCUMENT_SEARCH_SERVICE',
        '{
            "query": "sustainability in 2024 compared to 2025",
            "columns": ["SUMMARY", "FILE_NAME"],
            "limit": 15
        }'
    )
)['results'] AS RESULTS;


-- Next steps: build an entire SQL procedure to automatically process 
-- new stream of files:
-- https://blogs.kameshs.dev/i-built-a-document-search-service-in-30-seconds-and-you-can-too-with-snowflake-cortex-231f83edd22d

