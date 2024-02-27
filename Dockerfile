FROM inseefrlab/onyxia-python-pytorch:py3.11.6

ENV TIMEOUT=300
# set api as the current work dir
WORKDIR /api

# copy the requirements list
COPY requirements.txt requirements.txt

# install all the requirements
RUN mamba install -c conda-forge gdal -y &&\
    export PROJ_LIB=/opt/mamba/share/proj &&\
    pip install --no-cache-dir --upgrade -r requirements.txt

# copy the main code of fastapi
COPY ./app /api/app

# launch the unicorn server to run the api
# If you are running your container behind a TLS Termination Proxy (load balancer) like Nginx or Traefik,
# add the option --proxy-headers, this will tell Uvicorn to trust the headers sent by that proxy telling it
# that the application is running behind HTTPS, etc.
CMD ["uvicorn", "app.main:app",  "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--timeout-graceful-shutdown", "300"]
