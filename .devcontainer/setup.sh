#!/usr/bin/env bash
set -euo pipefail

echo "Installing OpenCode and agent-browser..."
npm install -g opencode-ai agent-browser

echo "Installing Chromium..."
sudo apt-get update
sudo apt-get install -y chromium

echo "Adding the agent-browser skill for OpenCode..."
npx -y skills add vercel-labs/agent-browser -a opencode -y

echo "Installing UI UX Pro Max CLI and skill for OpenCode..."
npm install -g ui-ux-pro-max-cli
uipro init --ai opencode

echo "Verifying the installs..."
opencode --version
agent-browser doctor
python3 --version

echo "Setup complete."