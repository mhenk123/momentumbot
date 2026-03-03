#!/bin/sh
# Entrypoint: initialisiert das data/-Verzeichnis beim ersten Start

mkdir -p /app/data

# config.json aus dem Image-Default kopieren, falls noch keine vorhanden
if [ ! -f /app/data/config.json ]; then
    echo "Keine config.json gefunden – kopiere Standard-Config..."
    cp /app/config.default.json /app/data/config.json
fi

exec python -u bot.py
