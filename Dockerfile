FROM inseefrlab/onyxia-python-pytorch:py3.12.9

ENV TIMEOUT=3600
ENV PROJ_LIB=tmp

# set app as the current work dir
WORKDIR /app

# copy the main code of fastapi
ADD . /app

# install all the requirements
RUN uv sync --frozen

# Gdal need to know which proj.db to use
RUN export PROJ_LIB=$(uv run python -c "from osgeo import __file__ as f; import os; print(os.path.join(os.path.dirname(f), 'data', 'proj'))")

# Expose port 5000
EXPOSE 5000

CMD ["uv", "run", "uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "5000", "--timeout-graceful-shutdown", "3600"]
