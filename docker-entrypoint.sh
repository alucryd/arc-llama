#!/bin/bash
set -e

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/arc-llama"
CONFIG_FILE="$CONFIG_DIR/config.toml"

# If no config exists and the user asked for 'serve', run init first
if [[ ! -f "$CONFIG_FILE" && "$1" == "serve" ]]; then
    echo "No config found at $CONFIG_FILE — running arc-llama init ..."
    arc-llama init --llama-server "${ARC_LLAMA_SERVER:-/usr/local/bin/llama-server}" --scan-path /models
fi

exec arc-llama "$@"
