FROM python:3.11-slim

WORKDIR /app

# Zuerst nur requirements kopieren (besseres Layer-Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bot und Standard-Config ins Image kopieren
COPY bot.py .
COPY entrypoint.sh .
# config.json als unveränderter Default – das entrypoint.sh kopiert sie
# beim ersten Start nach data/config.json (persistentes Volume)
COPY config.json config.default.json

RUN chmod +x entrypoint.sh

# data/ wird als Volume gemountet – kein Datei-Mount mehr nötig
VOLUME ["/app/data"]

ENTRYPOINT ["./entrypoint.sh"]
