#!/bin/bash

# Stop on error
set -e

echo "Starting setup script for Ubuntu..."

if [ "$(id -u)" -eq 0 ]; then
    echo "Running as root, skipping sudo..."
    SUDO=""
else
    SUDO="sudo"
fi

echo "Installing system packages..."
$SUDO apt-get update -y
$SUDO apt-get install -y tmux vim curl git

curl -LsSf https://astral.sh/uv/install.sh | sh

if [[ "$TERM" == "xterm-ghostty" ]]; then
    echo "Ghostty terminal detected. Applying compatibility fix..."
    echo "export TERM=xterm-256color" >> ~/.bashrc
    export TERM=xterm-256color
    echo "Added 'export TERM=xterm-256color' to ~/.bashrc"
fi

echo "Setup complete!"
echo "Run 'source \$HOME/.cargo/env' to activate uv in this shell."
