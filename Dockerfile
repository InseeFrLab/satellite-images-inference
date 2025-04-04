FROM inseefrlab/onyxia-python-pytorch:py3.12.9

ENV TIMEOUT=3600

# set app as the current work dir
WORKDIR /app

# copy the main code of fastapi
ADD . /app

# install all the requirements
RUN uv sync --frozen

CMD ["uv", "run", "uvicorn", "app.main:app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "3600"]
