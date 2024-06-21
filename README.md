# Computational Methods for Correcting Metal-Induced Reconstruction Artifacts in X-ray CT of Historic Gold Treasure

Author: David Diamond Wang Johansen (s214743)

## Overview

This GitHub repository contains the code developed for my BSc thesis, completed in the fall semester of 2024.

**Note:** Not all notebooks and intermediary investigations that were part of the process are included in this repository.

## Dataset Information

The Vindelev dataset is currently being prepared for a DOI. For inquiries related to the dataset, please contact:

- Carsten Gundlach: [cagu@fysik.dtu.dk](mailto:cagu@fysik.dtu.dk)
- Hans Martin Kjer: [hmkj@dtu.dk](mailto:hmkj@dtu.dk)

It is assumed that all the center slice data with COR corrections are saved under the `centres/` directory as `AcquisitionData` objects in `pickle` data format.

## Environment Setup

The Python virtual environment manager used for this project is `conda`. All the required channels, packages, and dependencies are listed in the `requirements.yml` file.

## Repository Structure

### `core.py`

This file contains the major functions and classes used throughout the project, including those required by `plotting.py`.

### `plotting.py`

This file contains the code to generate the majority of the plots, primarily related to 2D reconstructions, for the report.

### `notebooks/`

This directory contains Jupyter Notebooks used for some of the processing and for the 3D plotting.
