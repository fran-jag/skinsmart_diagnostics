FROM python:3.12-slim
LABEL author="MarcelSonne"

# install dependnecies.
RUN apt-get update && apt-get install -y \
procps \
libpq-dev \
build-essential \
# Install additional packages for building Python packages
cmake \
g++ \
&& pip install --upgrade pip setuptools wheel \
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*

# create working directory
WORKDIR /app

# copy requirements.txt into /app and install requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# copy
COPY . /app/

# CMD ["python", "main.py"]
CMD ["python"]