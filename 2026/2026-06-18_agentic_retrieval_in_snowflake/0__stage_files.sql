----------------------------------------------------------------------
-- FILE INGESTION INTO SNOWFLAKE (DEV)
-- Two options: Internal stage (upload files) or External stage (Azure)
-- Using DEV_SANDBOX.ABRETHO__RAW_FILES (shared DB, personal schema)
-- Note that one can't leverage personnal database.
----------------------------------------------------------------------

USE ROLE DNA_INF_DATA_SCIENCE;
USE WAREHOUSE DNA_DEFAULT_WH;

----------------------------------------------------------------------
-- OPTION 1: INTERNAL STAGE (for uploading files directly)
----------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS DEV_SANDBOX.ABRETHO__RAW_FILES;

CREATE OR REPLACE STAGE DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Internal stage for raw document ingestion';

-- UPLOADING FILES FROM SNOWSIGHT UI:
--   1. In the left sidebar, navigate to Data > Databases > DEV_SANDBOX > ABRETHO__RAW_FILES > Stages
--   2. Click on the RAW_FILES_UPLOAD stage
--   3. Click the "+ Files" button in the top-right corner
--   4. Drag and drop files or click "Browse" to select files from your machine
--   5. Click "Upload" to confirm
--
-- Alternatively, via SnowSQL CLI:
--   PUT file:///local/path/to/files/* @DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD;



-- Refresh directory table metadata after upload
ALTER STAGE DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD REFRESH;

-- Verify uploaded files
SELECT *
FROM DIRECTORY(@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD);

-- Count files
SELECT COUNT(*) AS FILE_COUNT
FROM DIRECTORY(@DEV_SANDBOX.ABRETHO__RAW_FILES.RAW_FILES_UPLOAD);


----------------------------------------------------------------------
-- OPTION 2: EXTERNAL STAGE (Azure Storage Account)
----------------------------------------------------------------------

-- Step 1: Create storage integration (requires ACCOUNTADMIN)
CREATE OR REPLACE STORAGE INTEGRATION AZURE_DOCS_INT
    TYPE = EXTERNAL_STAGE
    STORAGE_PROVIDER = 'AZURE'
    ENABLED = TRUE
    AZURE_TENANT_ID = '<your-azure-tenant-id>'
    STORAGE_ALLOWED_LOCATIONS = ('azure://<account>.blob.core.windows.net/<container>/<path>/');

-- Step 2: Get consent URL and app name, then grant access in Azure portal
DESC STORAGE INTEGRATION AZURE_DOCS_INT;

-- Step 3: Create the external stage
CREATE OR REPLACE STAGE DEV_SANDBOX.ABRETHO__RAW_FILES.EXTERNAL_DOCS
    STORAGE_INTEGRATION = AZURE_DOCS_INT
    URL = 'azure://<account>.blob.core.windows.net/<container>/<path>/'
    DIRECTORY = (ENABLE = TRUE);

-- Refresh directory table metadata
ALTER STAGE DEV_SANDBOX.ABRETHO__RAW_FILES.EXTERNAL_DOCS REFRESH;

-- Verify files from external storage
SELECT *
FROM DIRECTORY(@DEV_SANDBOX.ABRETHO__RAW_FILES.EXTERNAL_DOCS);

