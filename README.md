# 🛰️ Satellite Image Segmentation Inference

This repository contains code for performing segmentation inference on satellite images using deep learning models.

## Table of Contents

- [🛰️ Satellite Image Segmentation Inference](#️-satellite-image-segmentation-inference)
  - [Table of Contents](#table-of-contents)
  - [🚀 Introduction](#-introduction)
  - [🛠️ Usage](#️-usage)
    - [🔗 Step 1: Register new images](#-step-1-register-new-images)
    - [🧠 Step 2: Run inference via API](#-step-2-run-inference-via-api)
  - [🌐 API](#-api)
  - [🤖 Automation](#-automation)
  - [📜 License](#-license)

## 🚀 Introduction

This project is designed to perform segmentation inference on satellite images using deep learning models. The repository includes scripts for making predictions on satellite images and utilities for image format conversions. 📷🛰️

## 🛠️ Usage

To perform inference on a new set of satellite images stored in the S3 Bucket: 
📂 Path: `projet-slums-detection/data-raw/PLEIADES/<dep>/<year>/`

### 🔗 Step 1: Register new images

Before running inference, you must register the new images by linking them to their corresponding geometry polygons stored in the partitioned Parquet file:

📂 Path: `projet-slums-detection/data-raw/PLEIADES/filename-to-polygons/`

Run the following command:

```bash
python -m src.build_filename_to_polygons
```

### 🧠 Step 2: Run inference via API

Once registered, you can run inference on these new images using the API:

```bash
python -m src.make_predictions_from_api --dep <dep> --year <year>
```

## 🌐 API

All API-related code is in the app/ folder, built using FastAPI ⚡. The key files include:

- main.py 📌: Defines the API endpoints.
- utils.py 🔧: Contains utility functions for API operations.

## 🤖 Automation

The `argo-workflows/` folder contains templates that enable automation and parallelization of inference across multiple departments and years. ⚡🔄

## 📜 License

This project is licensed under the MIT License. 📄✅