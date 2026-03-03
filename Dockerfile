FROM python:3.11-slim

# Arbeitsverzeichnis im Container
WORKDIR /app

# Zuerst nur requirements kopieren (besseres Layer-Caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Restliche Projektdateien kopieren
COPY bot.py .
COPY config.json .

# .env wird NICHT ins Image kopiert – wird zur Laufzeit als Volume eingehängt
# database.db und bot.log entstehen erst zur Laufzeit

CMD ["python", "-u", "bot.py"]
