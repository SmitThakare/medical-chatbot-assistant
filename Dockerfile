FROM python:3.10-slim

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose Chainlit’s default port
EXPOSE 7860

# Run Chainlit app
CMD ["chainlit", "run", "app.py", "-h", "0.0.0.0", "-p", "7860"]