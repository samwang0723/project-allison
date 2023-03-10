FROM python:3.11-slim-bullseye as builder

ENV PATH="/root/.cargo/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends curl gcc &&  curl https://sh.rustup.rs -sSf | sh -s -- -y && apt-get install --reinstall libc6-dev -y

COPY requirements.txt .

RUN pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org


FROM python:3.11-slim-bullseye
# Copy pre-built packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/

# Set the working directory to /app
WORKDIR /app

# Copy the rest of the application code into the container at /app
COPY . /app

# Run the command to start the application
CMD ["python", "-i", "-t", "knowledge_base.py"]
