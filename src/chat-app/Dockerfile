# From https://techcommunity.microsoft.com/t5/fasttrack-for-azure/create-an-azure-openai-langchain-chromadb-and-chainlit-chat-app/ba-p/3885602

# # Stage 1 - Install build dependencies

# A Dockerfile must start with a FROM instruction that sets the base image for the container.
# The Python images come in many flavors, each designed for a specific use case.
# The python:3.11-slim image is a good base image for most applications.
# It is a minimal image built on top of Debian Linux and includes only the necessary packages to run Python.
# The slim image is a good choice because it is small and contains only the packages needed to run Python.
# For more information, see: 
# * https://hub.docker.com/_/python 
# * https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
FROM python:3.10-slim-bookworm AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PNPM_HOME=/pnpm

RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  software-properties-common \
  git \
  sudo \
  gpg \
  wget \
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:/pnpm/:$PATH"

# RUN wget -qO- https://get.pnpm.io/install.sh | ENV="$HOME/.bashrc" SHELL="$(which bash)" bash -
# RUN wget -qO- https://get.pnpm.io/install.sh | ENV="$HOME/.shrc" SHELL="$(which sh)" sh -
# RUN which pnpm

# RUN pnpm --version

# RUN wget -qO- https://deb.nodesource.com/setup_22.x | ENV="$HOME/.shrc" SHELL="$(which sh)" sh -

#   RUN apt-get install -y nodejs
# RUN node -v 

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Stage 2 - Copy only necessary files to the runner stage

# The FROM instruction initializes a new build stage for the application
FROM python:3.10-slim-bookworm

# Define the filename to copy as an argument
ARG FILENAME

# Deefine the port to run the application on as an argument
ARG PORT=8000

# Set an environment variable
ENV FILENAME=${FILENAME}

# Sets the working directory to /app
WORKDIR /app

RUN apt-get update

# RUN apt install -y curl

# RUN curl -sSL -O "https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb"
# RUN dpkg -i packages-microsoft-prod.deb
# RUN apt-get update
# RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17
# # optional: for bcp and sqlcmd
# RUN ACCEPT_EULA=Y apt-get install -y mssql-tools
# RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
# # optional: for unixODBC development headers
# RUN apt-get install -y unixodbc-dev

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"

# Copy the necessary files to the runner stage
COPY *.py .
COPY public public
COPY chainlit.md .
COPY schema.sql .
COPY ./.chainlit ./.chainlit

# Copy common Python dependencies from the itsarag directory 
# Requires `cp -r ../../lib/its_a_rag ./` before running `docker build .`	
COPY its_a_rag its_a_rag

#test file
COPY cat.jpg .
COPY pdfname.pdf .

COPY element.py /opt/venv/lib/python3.10/site-packages/chainlit/
COPY emitter.py /opt/venv/lib/python3.10/site-packages/chainlit/

# The EXPOSE instruction informs Docker that the container listens on the specified network ports at runtime.
# For more information, see: https://docs.docker.com/engine/reference/builder/#expose
EXPOSE 8000

# The ENTRYPOINT instruction has two forms:
# * ENTRYPOINT ["executable", "param1", "param2"] (exec form, preferred)
# * ENTRYPOINT command param1 param2 (shell form)
# The ENTRYPOINT instruction allows you to configure a container that will run as an executable.
# For more information, see: https://docs.docker.com/engine/reference/builder/#entrypoint
CMD ["chainlit", "run", "app.py", "--port", "8000", "--host", "0.0.0.0" ]
