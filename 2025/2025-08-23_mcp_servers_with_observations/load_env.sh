#!/bin/bash

# Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: This script must be sourced. Use 'source ./load_env.sh' or '. ./load_env.sh'."
  return
fi

# if ~/.env exists, export all env in the terminal
if [ -f ~/.env ]; then
  export $(grep -v '^#' ~/.env | xargs)
else
  echo "~/.env does not exist. Create it before running this script."
  return
fi

# Check if API_VERSION is set
if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
  echo "Error: Please check your '~/.env' file."
  return
else
  echo "Loaded."
fi
