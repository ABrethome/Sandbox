USE ROLE DNA_INF_DATA_SCIENCE;
USE WAREHOUSE DNA_DEFAULT_WH;

----------------------------------------------------------------------
-- STEP 1: CREATE CORTEX AGENT LINKED TO CORTEX SEARCH SERVICES
----------------------------------------------------------------------

-- TODO: not yet explored: add metadata in a table to add a metadata search based on structured queries.

CREATE OR REPLACE AGENT DEV_SANDBOX.ABRETHO__RAW_FILES.DOCUMENT_AGENT
  COMMENT = 'Test Agent for querying ingested documents via Cortex Search'
  PROFILE = '{"display_name": "Document Assistant", "color": "blue"}'
  FROM SPECIFICATION
  $$
  models:
    orchestration: auto

  orchestration:
    budget:
      seconds: 90
      tokens: 24000

  instructions:
    response: "You are a helpful document assistant. Answer questions based on the content retrieved from the document search tools. Always cite the source file name when providing information. If the answer is not found in the retrieved documents, say so clearly."
    orchestration: |
      Follow this search strategy to find the most relevant information:

      1. START with SummarySearch to identify which documents are most relevant to the user's question. This searches document-level summaries to narrow down by topic.

      2. Once you have identified relevant document(s) from the summary results, use FullDocumentSearch with a FILE_NAME filter to search for specific chunks within those documents. This gives you the detailed content needed to answer the question.

      3. FALLBACK: If SummarySearch returns no relevant results, use FullDocumentSearch without a filter to search across all document chunks directly. From those results, identify the relevant FILE_NAME(s), then search again with the FILE_NAME filter to retrieve all relevant chunks from those specific documents.

      Always prefer targeted searches (with FILE_NAME filter) over broad searches when possible, as they return more precise results.
    sample_questions:
      - question: "What are the key sustainability goals?"
      - question: "Summarize the heavy duty motor oils document"
      - question: "What does the report say about emissions?"

  tools:
    - tool_spec:
        type: "cortex_search"
        name: "SummarySearch"
        description: "Searches document-level summaries to identify which documents are relevant to a topic. Use this FIRST to narrow down which documents to look into. Returns document summaries with file names."
    - tool_spec:
        type: "cortex_search"
        name: "FullDocumentSearch"
        description: "Searches through detailed document chunks (paragraphs, tables, image descriptions). Use this AFTER identifying relevant documents via SummarySearch, filtering by FILE_NAME to get specific content. Can also be used without filter as a fallback for broad searches."

  tool_resources:
    SummarySearch:
      name: "DEV_SANDBOX.ABRETHO__RAW_FILES.SUMMARY_DOCUMENT_SEARCH_SERVICE"
      max_results: "5"
      title_column: "FILE_NAME"
      columns_and_descriptions:
        SUMMARY:
          description: "A high-level summary of the entire document. Use to determine if a document is relevant to the user's question."
          type: "string"
          searchable: true
          filterable: false
        FILE_NAME:
          description: "The name of the source document file (e.g. 'report.pdf', 'presentation.pptx'). Use the returned file names to filter FullDocumentSearch."
          type: "string"
          searchable: false
          filterable: true
    FullDocumentSearch:
      name: "DEV_SANDBOX.ABRETHO__RAW_FILES.FULL_DOCUMENT_SEARCH_SERVICE"
      max_results: "15"
      title_column: "FILE_NAME"
      columns_and_descriptions:
        CHUNK_TEXT:
          description: "The text content of the document chunk. Contains paragraphs, table summaries, and image descriptions from parsed documents."
          type: "string"
          searchable: true
          filterable: false
        FILE_NAME:
          description: "The name of the source document file. Use as a filter to search within a specific document identified by SummarySearch."
          type: "string"
          searchable: false
          filterable: true
  $$;

-- Grant usage to allow other roles to use the agent
GRANT USAGE ON AGENT DEV_SANDBOX.ABRETHO__RAW_FILES.DOCUMENT_AGENT TO ROLE DNA_SNOWFLAKE_USERS;

-- Verify the agent was created
DESCRIBE AGENT DEV_SANDBOX.ABRETHO__RAW_FILES.DOCUMENT_AGENT;

-- go use newly created agent:
-- https://ai.snowflake.com/infineum/infineum/#/agents