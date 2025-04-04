FROM inseefrlab/onyxia-python-pytorch:py3.12.9

ENV TIMEOUT=3600

# set api as the current work dir
WORKDIR /api

# copy the main code of fastapi
COPY ./app /api/app

# install all the requirements
RUN uv sync --frozen

CMD ["uv", "run", "uvicorn", "app.main:app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "3600"]
