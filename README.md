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

## Getting started

```
git clone https://github.com/InseeFrLab/satellite-images-inference.git
cd satellite-images-inference/
uv sync
export PROJ_LIB=$(uv run python -c "from osgeo import __file__ as f; import os; print(os.path.join(os.path.dirname(f), 'data', 'proj'))")
```

## Collect new PLEIADES images

1. Open a VSCode instance with maximum persistence.

2. Modify the `bash/download_pleiades_ign.sh` script with the server-specific FTP information and the required arguments (department, year).

3. Run the script with:

```bash
source ./bash/download_pleiades_ign.sh
```

4. Convert the JP2 images to TIF using:
```bash
uv run -m src.write_jp2_to_tiff --folder_path <local_folder>
```

5. Copy the new images to S3, following this structure:
`projet-slums-detection/data-raw/PLEIADES/<dep>/<year>/`

6. Create the data-roi geojson:
```{bash}
uv run -m src.build_data_roi --dep_code <Dep code>
```
For Corsica, enter '2A|2B' as dep_code, and the ROI for the entire island of Corsica will be retrieved.

7. Copy the geojson to S3, in this folder:
`projet-slums-detection/data-roi/`


## 🛠️ Usage

To perform inference on a new set of satellite images stored in the S3 Bucket: 

📂 Path: `projet-slums-detection/data-raw/PLEIADES/<dep>/<year>/`

### 🔗 Step 1: Register new images

Before running inference, you must register the new images by linking them to their corresponding geometry polygons stored in the partitioned Parquet file:

📂 Path: `projet-slums-detection/data-raw/PLEIADES/filename-to-polygons/`

Run the following command:

```bash
uv run -m src.build_filename_to_polygons
```

### 🧠 Step 2: Run inference via API

Once registered, you can run inference on these new images using the API:

```bash
uv run -m src.make_predictions_from_api --dep <dep> --year <year>
```

### Step 3: Make statistics on building area

Create the data-clusters parquet folder:
```{bash}
uv run -m src.build_data_clusters --dep_code <Dep code> --dep_name <Dep name>
```
For Corsica, enter '2A|2B' as dep_code, and the ROI for the entire island of Corsica will be retrieved.

```bash
uv run -m src.make_statistics_from_api --dep <dep> --year <year>
```

### Step 4: Constructions/destructions of buildings

```bash
uv run -m src.constructions_destructions --dep <dep> --year <year>
```

## 🌐 API

All API-related code is in the app/ folder, built using FastAPI ⚡. The key files include:

- main.py 📌: Defines the API endpoints.
- utils.py 🔧: Contains utility functions for API operations.

## 🤖 Automation

The `argo-workflows/` folder contains templates that enable automation and parallelization of inference across multiple departments and years. ⚡🔄

## 📜 License

This project is licensed under the MIT License. 📄✅
