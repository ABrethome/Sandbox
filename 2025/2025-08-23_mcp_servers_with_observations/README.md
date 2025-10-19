# Model Context Protocol server

This code is run on an Azure compute instance to leverage Azure OpenAI endpoint.

### Create conda env

```bash
conda env create -f environment.yaml
```

### Set env variables in `~/.env`
Set the following env variables ot be able to run the client and the server:
```
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_BASIC_DEPLOYMENT_NAME
AZURE_OPENAI_BASIC_MODEL_NAME
```

### Create a fake SQL database

Run notebook in `create_sql_database.ipynb`.

### 🚀 Run the MCP server locally
Open `config.yaml`, and replace values if needed.
Ensure your env variables for Azure OpenAI are set in `~/.env`.

Run the app with Uvicorn ASGI server:
```bash
source ./load_env.sh
conda activate sandbox
python server.py
```
Will open in browser at http://0.0.0.0:3002 (or http://localhost:3002/).

You can test the MCP server by using the basic client provided:

**In a new terminal**, run:

```bash
source ./load_env.sh
conda activate sandbox
python client.py --interactive
```

You will be able to choose a question and test that the MCP server correctly provides an answer to the basic agent in the client.
The basic client does not leverage observations here, and act as just any other agent out there without proper citation support.
