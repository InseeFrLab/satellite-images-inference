#!/bin/bash

uv sync
uv run pre-commit install
export MLFLOW_MODEL_NAME=Segmentation-multiclass
export MLFLOW_MODEL_VERSION="1"
unset AWS_SESSION_TOKEN
