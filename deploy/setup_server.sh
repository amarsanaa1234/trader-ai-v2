#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 droplet for crypto_ai. Run as root.
set -euo pipefail

echo "== 1/7 system packages"
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-venv python3-pip ufw fail2ban \
  unattended-upgrades curl sqlite3

echo "== 2/7 service user 'bot' (the bot never runs as root)"
id bot >/dev/null 2>&1 || adduser --disabled-password --gecos "" bot

echo "== 3/7 firewall: only SSH in; dashboard stays on 127.0.0.1"
ufw allow OpenSSH
ufw --force enable

echo "== 4/7 SSH: keys only, no passwords"
sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl reload ssh || systemctl reload sshd

echo "== 5/7 4GB swap (safety net for the AI model) + timezone"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
timedatectl set-timezone Australia/Sydney

echo "== 6/7 Ollama (local AI, official installer) + model"
if ! command -v ollama >/dev/null; then curl -fsSL https://ollama.com/install.sh | sh; fi
systemctl enable --now ollama
sleep 3
ollama pull qwen2.5:7b

echo "== 7/7 app folder"
mkdir -p /opt/trader_ai_v2
chown -R bot:bot /opt/trader_ai_v2
echo "SETUP DONE"
