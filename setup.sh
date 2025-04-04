#!/bin/bash

uv run pre-commit install
export MLFLOW_MODEL_NAME=Segmentation-multiclass
export MLFLOW_MODEL_VERSION="1"
