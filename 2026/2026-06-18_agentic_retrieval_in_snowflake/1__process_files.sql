----------------------------------------------------------------------
-- DOCUMENT PROCESSING PIPELINE
-- Step 0: Convert unsupported formats into supported formats
-- Step 1: Parse & extract (OCR + layout)
-- Step 2: Enrich (summarize tables, describe images via vision model)
-- Step 3: Clean (remove noise: page numbers, footers, etc.)
----------------------------------------------------------------------

USE ROLE DNA_INF_DATA_SCIENCE;
USE WAREHOUSE DNA_DEFAULT_WH;

----------------------------------------------------------------------
-- STEP 0: CONVERT UNSUPPORTED FORMATS TO SUPPORTED FORMATS
-- Converts XLSX files to HTML (one HTML file per sheet)
-- Supported by AI_PARSE_DOCUMENT: PDF, PPTX, DOCX, JPEG, PNG, TIFF, HTML, TXT
-- this approach would have to be reproduced for each unsupported format by default
----------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE DEV_SANDBOX.ABRETHO__RAW_FILES.CONVERT_UNSUPPORTED_FILES()
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python', 'openpyxl', 'pandas')
HANDLER = 'run'
AS
$$
import pandas as pd
from snowflake.snowpark.files import SnowflakeFile
import io

def run(session):
    stage_path = '@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD'

    existing_files = session.sql(f"SELECT RELATIVE_PATH FROM DIRECTORY({stage_path}) WHERE RELATIVE_PATH ILIKE '%.html'").collect()
    existing_html = set(row['RELATIVE_PATH'] for row in existing_files)

    files_df = session.sql(f"SELECT RELATIVE_PATH FROM DIRECTORY({stage_path}) WHERE RELATIVE_PATH ILIKE '%.xlsx' OR RELATIVE_PATH ILIKE '%.xls'").collect()

    converted_count = 0
    skipped_count = 0

    for row in files_df:
        relative_path = row['RELATIVE_PATH']
        base_name = relative_path.rsplit('.', 1)[0]
        expected_html_prefix = f"{base_name}__"

        if any(h.startswith(expected_html_prefix) and h.endswith('.html') for h in existing_html):
            skipped_count += 1
            continue

        file_url = session.sql(f"SELECT BUILD_SCOPED_FILE_URL('{stage_path}', '{relative_path}')").collect()[0][0]

        with SnowflakeFile.open(file_url, 'rb') as f:
            xlsx_data = io.BytesIO(f.read())

        xl = pd.ExcelFile(xlsx_data, engine='openpyxl')

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            html_content = df.to_html(index=False, na_rep='', border=0)
            html_with_meta = f"<html><head><title>{sheet_name}</title></head><body><h1>{sheet_name}</h1>{html_content}</body></html>"

            safe_sheet = sheet_name.replace(' ', '_').replace('/', '_')
            html_filename = f"{base_name}__{safe_sheet}.html"

            session.file.put_stream(
                io.BytesIO(html_with_meta.encode('utf-8')),
                f"{stage_path}/{html_filename}",
                auto_compress=False,
                overwrite=True
            )
            converted_count += 1

    session.sql(f"ALTER STAGE DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD REFRESH").collect()
    return f"Converted {converted_count} sheets from {len(files_df) - skipped_count} new Excel file(s). Skipped {skipped_count} already-processed file(s)."
$$;

CALL DEV_SANDBOX.ABRETHO__RAW_FILES.CONVERT_UNSUPPORTED_FILES();

ALTER STAGE DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD REFRESH;

-- visualize files
SELECT *
FROM DIRECTORY(@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD);


----------------------------------------------------------------------
-- STEP 1: PARSE DOCUMENTS (text + layout + images)
----------------------------------------------------------------------

-- drop table for testing
--DROP TABLE DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS;

-- cache table for raw AI_PARSE_DOCUMENT output (avoids re-parsing the same doc)
CREATE TABLE IF NOT EXISTS DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_RAW_OUTPUT (
    FILE_PATH        VARCHAR,
    FILE_NAME        VARCHAR,
    RAW_RESULT       VARIANT,
    PARSED_AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- dedicated table for markdown parsing
CREATE TABLE IF NOT EXISTS DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS (
    FILE_PATH        VARCHAR,
    FILE_NAME        VARCHAR,
    PAGE_NUMBER      INTEGER,
    RAW_CONTENT      VARCHAR,
    PARSED_AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- dedicated table for image processing
CREATE TABLE IF NOT EXISTS DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_IMAGES (
    FILE_PATH        VARCHAR,
    FILE_NAME        VARCHAR,
    IMAGE_ID         VARCHAR,
    IMAGE_BASE64     VARCHAR,
    PARSED_AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- PDF, PPTX, DOCX: parse once and cache the raw result
-- Limitations: how to decide when to run on OCR mode?
INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_RAW_OUTPUT
    (FILE_PATH, FILE_NAME, RAW_RESULT)
SELECT
    d.RELATIVE_PATH                                         AS FILE_PATH,
    SPLIT_PART(d.RELATIVE_PATH, '/', -1)                    AS FILE_NAME,
    parsed.result                                           AS RAW_RESULT
FROM
    DIRECTORY(@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD) d,
    LATERAL (
        SELECT AI_PARSE_DOCUMENT(
            TO_FILE('@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD', d.RELATIVE_PATH),
            {'mode': 'LAYOUT', 'page_split': true, 'extract_images': true}
        ) AS result
    ) parsed
WHERE d.RELATIVE_PATH NOT IN (
    SELECT DISTINCT FILE_PATH FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_RAW_OUTPUT
)
AND (d.RELATIVE_PATH ILIKE '%.pdf'
  OR d.RELATIVE_PATH ILIKE '%.pptx'
  OR d.RELATIVE_PATH ILIKE '%.docx');

-- extract pages from cached raw output into PARSED_DOCUMENTS
INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
    (FILE_PATH, FILE_NAME, PAGE_NUMBER, RAW_CONTENT)
SELECT
    r.FILE_PATH,
    r.FILE_NAME,
    page.VALUE:index::INTEGER                               AS PAGE_NUMBER,
    page.VALUE:content::VARCHAR                             AS RAW_CONTENT
FROM
    DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_RAW_OUTPUT r,
    LATERAL FLATTEN(input => r.RAW_RESULT:pages) page
WHERE (r.FILE_PATH ILIKE '%.pdf' OR r.FILE_PATH ILIKE '%.pptx' OR r.FILE_PATH ILIKE '%.docx')
AND r.FILE_PATH NOT IN (
    SELECT DISTINCT FILE_PATH FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
);

-- extract images from cached raw output into PARSED_IMAGES
INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_IMAGES
    (FILE_PATH, FILE_NAME, IMAGE_ID, IMAGE_BASE64)
SELECT
    r.FILE_PATH,
    r.FILE_NAME,
    img.VALUE:id::VARCHAR                                   AS IMAGE_ID,
    img.VALUE:image_base64::VARCHAR                         AS IMAGE_BASE64
FROM
    DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_RAW_OUTPUT r,
    LATERAL FLATTEN(input => r.RAW_RESULT:pages) page,
    LATERAL FLATTEN(input => page.VALUE:images) img
WHERE r.FILE_PATH NOT IN (
    SELECT DISTINCT FILE_PATH FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_IMAGES
);

-- HTML and TXT: cache raw output
INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_RAW_OUTPUT
    (FILE_PATH, FILE_NAME, RAW_RESULT)
SELECT
    d.RELATIVE_PATH                                         AS FILE_PATH,
    SPLIT_PART(d.RELATIVE_PATH, '/', -1)                    AS FILE_NAME,
    parsed.result                                           AS RAW_RESULT
FROM
    DIRECTORY(@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD) d,
    LATERAL (
        SELECT AI_PARSE_DOCUMENT(
            TO_FILE('@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD', d.RELATIVE_PATH),
            {'mode': 'LAYOUT'}
        ) AS result
    ) parsed
WHERE (d.RELATIVE_PATH ILIKE '%.html' OR d.RELATIVE_PATH ILIKE '%.txt')
AND d.RELATIVE_PATH NOT IN (
    SELECT DISTINCT FILE_PATH FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_RAW_OUTPUT
);

-- HTML and TXT: single page (no page_split support)
INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
    (FILE_PATH, FILE_NAME, PAGE_NUMBER, RAW_CONTENT)
SELECT
    r.FILE_PATH,
    r.FILE_NAME,
    0                                                       AS PAGE_NUMBER,
    r.RAW_RESULT:content::VARCHAR                           AS RAW_CONTENT
FROM
    DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_RAW_OUTPUT r
WHERE (r.FILE_PATH ILIKE '%.html' OR r.FILE_PATH ILIKE '%.txt')
AND r.FILE_PATH NOT IN (
    SELECT DISTINCT FILE_PATH FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
);

-- JPEG, PNG, TIFF: image-to-text via multi-modal AI_COMPLETE
-- cost optimisation possible?
INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
    (FILE_PATH, FILE_NAME, RAW_CONTENT)
SELECT
    d.RELATIVE_PATH                                         AS FILE_PATH,
    SPLIT_PART(d.RELATIVE_PATH, '/', -1)                    AS FILE_NAME,
    AI_COMPLETE(
        'claude-4-sonnet',
        'Describe this image in detail. Include what it depicts, any text visible, data shown in charts/graphs, and its likely purpose.',
        TO_FILE('@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD', d.RELATIVE_PATH)
    )                                                       AS RAW_CONTENT
FROM DIRECTORY(@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD) d
WHERE (d.RELATIVE_PATH ILIKE '%.jpeg'
    OR d.RELATIVE_PATH ILIKE '%.jpg'
    OR d.RELATIVE_PATH ILIKE '%.png'
    OR d.RELATIVE_PATH ILIKE '%.tiff'
    OR d.RELATIVE_PATH ILIKE '%.tif')
AND d.RELATIVE_PATH NOT IN (
    SELECT DISTINCT FILE_PATH FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
);

-- list file by filename
SELECT DISTINCT FILE_NAME, PAGE_NUMBER, RAW_CONTENT
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
ORDER BY FILE_NAME;

SELECT COUNT(DISTINCT FILE_NAME) AS UNIQUE_FILE_COUNT
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS;

-- visualize 1 specific filename
SELECT *
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
WHERE FILE_NAME = 'Heavy_Duty_Motor_Oils_PoC.pptx';

-- identify the missing files, not being processed
SELECT SPLIT_PART(RELATIVE_PATH, '/', -1) AS FILE_NAME
FROM DIRECTORY(@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD)
WHERE SPLIT_PART(RELATIVE_PATH, '/', -1) NOT IN (
    SELECT DISTINCT FILE_NAME FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
);

-- Diagnose why files failed parsing (returns error details instead of NULL)
-- Note: .xlsx is NOT supported by AI_PARSE_DOCUMENT (only PDF, PPTX, DOCX, JPEG, PNG, TIFF, HTML, TXT)
SELECT
    d.RELATIVE_PATH AS FILE_NAME,
    parsed.result:error::VARCHAR AS ERROR_MESSAGE,
    parsed.result:metadata AS METADATA
FROM DIRECTORY(@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD) d,
    LATERAL (
        SELECT AI_PARSE_DOCUMENT(
            TO_FILE('@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD', d.RELATIVE_PATH),
            {'mode': 'LAYOUT'},
            TRUE
        ) AS result
    ) parsed
WHERE SPLIT_PART(d.RELATIVE_PATH, '/', -1) IN (
    '2023-sustainability-report.pdf'
);
-- internal error obtained for '2023-sustainability-report.pdf'. Unclear why. Unclear how to debug this within Snowflake.
-- is that a concern? does this happen frequently? Or am I highly unlikely to have got this issue with only 10 files? 
-- what will happen with thousands of files?

----------------------------------------------------------------------
-- STEP 2: ENRICH (summarize tables, describe images via vision model)
----------------------------------------------------------------------

-- list files containing an image
SELECT FILE_NAME, IMAGE_ID
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_IMAGES

-- 2a: Save extracted images to stage, then describe them with multimodal AI_COMPLETE
-- one must save the images to leverage AI_COMPLETE with TO_FILE

CREATE OR REPLACE STAGE DEV_SANDBOX.ABRETHO__RAW_FILES.EXTRACTED_IMAGES
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

CREATE OR REPLACE PROCEDURE DEV_SANDBOX.ABRETHO__RAW_FILES.SAVE_IMAGES_TO_STAGE()
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'run'
AS
$$
import base64
import io
import os

def run(session):
    rows = session.sql("""
        SELECT FILE_PATH, FILE_NAME, IMAGE_ID, IMAGE_BASE64
        FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_IMAGES
        WHERE IMAGE_BASE64 IS NOT NULL
    """).collect()

    count = 0
    for row in rows:
        file_name = os.path.splitext(row['FILE_NAME'])[0]
        image_id = row['IMAGE_ID']
        image_base64 = row['IMAGE_BASE64']
        if ',' in image_base64:
            image_base64 = image_base64.split(',', 1)[1]
        image_bytes = base64.b64decode(image_base64)
        stage_path = f"@DEV_SANDBOX.ABRETHO__RAW_FILES.EXTRACTED_IMAGES/{file_name}/{image_id}"
        session.file.put_stream(
            io.BytesIO(image_bytes),
            stage_path,
            auto_compress=False,
            overwrite=True
        )
        count += 1

    session.sql("ALTER STAGE DEV_SANDBOX.ABRETHO__RAW_FILES.EXTRACTED_IMAGES REFRESH").collect()
    return f"Saved {count} images to stage."
$$;

CALL DEV_SANDBOX.ABRETHO__RAW_FILES.SAVE_IMAGES_TO_STAGE();


CREATE TABLE IF NOT EXISTS DEV_SANDBOX.ABRETHO__RAW_FILES.IMAGE_DESCRIPTIONS (
    FILE_PATH        VARCHAR,
    IMAGE_NAME       VARCHAR,
    IMAGE_DESCRIPTION VARCHAR,
    DESCRIBED_AT     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- transform image to text
INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.IMAGE_DESCRIPTIONS
    (FILE_PATH, IMAGE_NAME, IMAGE_DESCRIPTION)
SELECT *
FROM (
    SELECT
        p.FILE_PATH,
        p.IMAGE_ID AS IMAGE_NAME,
        AI_COMPLETE(
            'claude-4-sonnet',
            'Describe this image in detail. Include what it depicts, any text visible, data shown in charts/graphs, and its likely purpose in a technical document.',
            TO_FILE('@DEV_SANDBOX.ABRETHO__RAW_FILES.EXTRACTED_IMAGES', SPLIT_PART(p.FILE_NAME, '.', 1) || '/' || p.IMAGE_ID)
        ) AS IMAGE_DESCRIPTION
    FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_IMAGES p
    WHERE p.IMAGE_BASE64 IS NOT NULL
    AND p.IMAGE_ID NOT IN (
        SELECT DISTINCT IMAGE_NAME FROM DEV_SANDBOX.ABRETHO__RAW_FILES.IMAGE_DESCRIPTIONS
    )
) WHERE IMAGE_DESCRIPTION IS NOT NULL;


-- 2b: Enrich text content (summarize tables, inject image descriptions)
CREATE TABLE IF NOT EXISTS DEV_SANDBOX.ABRETHO__RAW_FILES.ENRICHED_DOCUMENTS (
    FILE_PATH        VARCHAR,
    FILE_NAME        VARCHAR,
    PAGE_NUMBER      INTEGER,
    ENRICHED_CONTENT VARCHAR,
    ENRICHED_AT      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Deterministic image replacement via procedure (SQL can't handle N distinct replacements per row)
CREATE OR REPLACE PROCEDURE DEV_SANDBOX.ABRETHO__RAW_FILES.REPLACE_IMAGE_REFERENCES()
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'run'
AS
$$
import re

def run(session):
    pages = session.sql("""
        SELECT FILE_PATH, FILE_NAME, PAGE_NUMBER, RAW_CONTENT
        FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
        WHERE RAW_CONTENT LIKE '%![%](%'
        AND (FILE_PATH, PAGE_NUMBER) NOT IN (
            SELECT FILE_PATH, PAGE_NUMBER FROM DEV_SANDBOX.ABRETHO__RAW_FILES.ENRICHED_DOCUMENTS
        )
    """).collect()

    descriptions = session.sql("""
        SELECT FILE_PATH, IMAGE_NAME, IMAGE_DESCRIPTION
        FROM DEV_SANDBOX.ABRETHO__RAW_FILES.IMAGE_DESCRIPTIONS
    """).collect()

    desc_lookup = {}
    for row in descriptions:
        key = (row['FILE_PATH'], row['IMAGE_NAME'])
        desc_lookup[key] = row['IMAGE_DESCRIPTION']

    img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    rows_to_insert = []
    for page in pages:
        file_path = page['FILE_PATH']
        file_name = page['FILE_NAME']
        page_number = page['PAGE_NUMBER']
        content = page['RAW_CONTENT']

        def replace_match(m):
            image_ref = m.group(2)
            key = (file_path, image_ref)
            desc = desc_lookup.get(key)
            if desc:
                return f"[Image: {desc}]"
            return m.group(0)

        enriched = img_pattern.sub(replace_match, content)
        rows_to_insert.append((file_path, file_name, page_number, enriched))

    if rows_to_insert:
        df = session.create_dataframe(
            rows_to_insert,
            schema=['FILE_PATH', 'FILE_NAME', 'PAGE_NUMBER', 'ENRICHED_CONTENT']
        )
        # Added column_order='name' to both save_as_table calls. 
        # This tells Snowpark to match columns by name rather than position, 
        # so the 5th column (ENRICHED_AT) will use its DEFAULT CURRENT_TIMESTAMP() 
        # value instead of requiring an explicit value from the dataframe.
        df.write.mode('append').save_as_table(
            'DEV_SANDBOX.ABRETHO__RAW_FILES.ENRICHED_DOCUMENTS',
            column_order='name'
        )

    pages_no_images = session.sql("""
        SELECT FILE_PATH, FILE_NAME, PAGE_NUMBER, RAW_CONTENT
        FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS
        WHERE RAW_CONTENT NOT LIKE '%![%](%'
        AND (FILE_PATH, PAGE_NUMBER) NOT IN (
            SELECT FILE_PATH, PAGE_NUMBER FROM DEV_SANDBOX.ABRETHO__RAW_FILES.ENRICHED_DOCUMENTS
        )
    """).collect()

    no_img_rows = [(r['FILE_PATH'], r['FILE_NAME'], r['PAGE_NUMBER'], r['RAW_CONTENT']) for r in pages_no_images]
    if no_img_rows:
        df2 = session.create_dataframe(
            no_img_rows,
            schema=['FILE_PATH', 'FILE_NAME', 'PAGE_NUMBER', 'ENRICHED_CONTENT']
        )
        df2.write.mode('append').save_as_table(
            'DEV_SANDBOX.ABRETHO__RAW_FILES.ENRICHED_DOCUMENTS',
            column_order='name'
        )

    return f"Enriched {len(rows_to_insert)} pages with image replacements, {len(no_img_rows)} pages passed through unchanged."
$$;

CALL DEV_SANDBOX.ABRETHO__RAW_FILES.REPLACE_IMAGE_REFERENCES();

-- view files to debug side-by-side
SELECT
    p.FILE_NAME,
    p.page_number,
    p.RAW_CONTENT AS PARSED_CONTENT,
    e.ENRICHED_CONTENT
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS p
LEFT JOIN DEV_SANDBOX.ABRETHO__RAW_FILES.ENRICHED_DOCUMENTS e
    ON p.FILE_PATH = e.FILE_PATH
    AND p.PAGE_NUMBER = e.PAGE_NUMBER
WHERE REGEXP_COUNT(p.RAW_CONTENT, '!\\[[^\\]]*\\]\\([^)]+\\)') >= 1


----------------------------------------------------------------------
-- STEP 3: CLEAN (remove noise: page numbers, footers, boilerplate)
----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS DEV_SANDBOX.ABRETHO__RAW_FILES.CLEANED_DOCUMENTS (
    FILE_PATH        VARCHAR,
    FILE_NAME        VARCHAR,
    PAGE_NUMBER      INTEGER,
    CLEANED_CONTENT  VARCHAR,
    CLEANED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Again, this is an expensive way of handling cleaning. Unsure this is workable long term cost-wise.
-- the current TDS pipeline does this kind of cleaning, but it first carries out cleaning deterministcly, and only to 
-- paragraph chunks, meaning it does not run this on EVERYTHING. 
-- Here, it is everything because of how AI Parse works on Snowflake,
-- which again feels like a token burner long term.
-- unsure if there is a cheaper way to clean those enriched documents, 
-- to make them easier to understand for a model.


INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.CLEANED_DOCUMENTS
    (FILE_PATH, FILE_NAME, PAGE_NUMBER, CLEANED_CONTENT)
SELECT
    FILE_PATH,
    FILE_NAME,
    PAGE_NUMBER,
    AI_COMPLETE(
        'claude-4-sonnet',
        CONCAT(
            'You are a text cleaning assistant. Given the following enriched document, remove all non-informative or repetitive elements while preserving all meaningful content.\n\n',
            'REMOVE the following types of content:\n',
            '- Page numbers (e.g. "Page 3 of 10", "- 3 -", standalone numbers at top/bottom)\n',
            '- Recurring headers/footers (e.g. "Confidential Information", company names repeated on every page, document IDs)\n',
            '- Watermarks or stamps text\n',
            '- Empty placeholder lines or excessive whitespace\n',
            'KEEP intact:\n',
            '- All substantive text, paragraphs, and headings\n',
            '- Table summaries and image descriptions\n',
            '- Captions and annotations that add meaning\n',
            '- Section titles, numbered lists, bullet points\n\n',
            'Return ONLY the cleaned content with no commentary or explanation.\n\n',
            '--- ENRICHED CONTENT ---\n',
            ENRICHED_CONTENT
        )
    ) AS CLEANED_CONTENT
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.ENRICHED_DOCUMENTS
WHERE (FILE_PATH, PAGE_NUMBER) NOT IN (
    SELECT FILE_PATH, PAGE_NUMBER FROM DEV_SANDBOX.ABRETHO__RAW_FILES.CLEANED_DOCUMENTS
);

-- check results
SELECT
    p.FILE_NAME,
    p.PAGE_NUMBER,
    p.RAW_CONTENT AS PARSED_CONTENT,
    e.ENRICHED_CONTENT,
    c.CLEANED_CONTENT
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.PARSED_DOCUMENTS p
LEFT JOIN DEV_SANDBOX.ABRETHO__RAW_FILES.ENRICHED_DOCUMENTS e
    ON p.FILE_PATH = e.FILE_PATH
    AND p.PAGE_NUMBER = e.PAGE_NUMBER
LEFT JOIN DEV_SANDBOX.ABRETHO__RAW_FILES.CLEANED_DOCUMENTS c
    ON p.FILE_PATH = c.FILE_PATH
    AND p.PAGE_NUMBER = c.PAGE_NUMBER


----------------------------------------------------------------------
-- STEP 4: Aggregate
-- this is key, as some paragraph are across pages, or even tables across pages.
-- never chunk based on pages, but rather on context ;-)
----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS DEV_SANDBOX.ABRETHO__RAW_FILES.FULL_DOCUMENTS (
    FILE_PATH VARCHAR,
    FILE_NAME VARCHAR,
    CONTENT   VARCHAR,
    AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.FULL_DOCUMENTS
    (FILE_PATH, FILE_NAME, CONTENT)
SELECT
    FILE_PATH,
    FILE_NAME,
    LISTAGG(CLEANED_CONTENT, '\n\n') WITHIN GROUP (ORDER BY PAGE_NUMBER) AS CONTENT
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.CLEANED_DOCUMENTS
WHERE FILE_PATH NOT IN (
    SELECT FILE_PATH FROM DEV_SANDBOX.ABRETHO__RAW_FILES.FULL_DOCUMENTS
)
GROUP BY FILE_PATH, FILE_NAME;

-- visualize output
SELECT FILE_NAME, CONTENT
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.FULL_DOCUMENTS;


----------------------------------------------------------------------
-- STEP 5: Summarize
----------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS DEV_SANDBOX.ABRETHO__RAW_FILES.SUMMARY_DOCUMENTS (
    FILE_PATH VARCHAR,
    FILE_NAME VARCHAR,
    SUMMARY   VARCHAR,
    AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

INSERT INTO DEV_SANDBOX.ABRETHO__RAW_FILES.SUMMARY_DOCUMENTS
    (FILE_PATH, FILE_NAME, SUMMARY)
SELECT
    FILE_PATH,
    FILE_NAME,
    AI_COMPLETE(
        'claude-4-sonnet',
        CONCAT(
            'Summarize the following document in a maximum of 1000 characters. The summary must:\n',
            '1. Clearly state what the document is about (topic, scope, key findings)\n',
            '2. Include the document creation date if mentioned anywhere in the text\n',
            '3. List the types of user queries this document would be useful for (e.g. "useful for questions about X, Y, Z")\n\n',
            'Be concise, informative, and factual. Do not exceed 1000 characters.\n\n',
            '--- DOCUMENT ---\n',
            CONTENT
        )
    ) AS SUMMARY
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.FULL_DOCUMENTS
WHERE FILE_PATH NOT IN (
    SELECT FILE_PATH FROM DEV_SANDBOX.ABRETHO__RAW_FILES.SUMMARY_DOCUMENTS
);


-- visualize output
SELECT FILE_NAME, SUMMARY
FROM DEV_SANDBOX.ABRETHO__RAW_FILES.SUMMARY_DOCUMENTS;
    