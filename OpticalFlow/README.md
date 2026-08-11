# Optical Flow Project

## Overview
This project implements Optical Flow in and Python using OpenCV.

## Features
- Computes dense and sparse optical flow
- Supports Lucas-Kanade and Farneback methods
- Python implementation with machine learning clustering

## Python Implementation - Optical Flow

### Dependencies
The Python implementation requires the following packages listed in `requirements.txt`:
- scikit-learn
- opencv-python
- numpy

### Installation
Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Execution
To run the Python optical flow implementation:

```bash
python opticalflow.py [video_path]
```

### Command Line Arguments
The Python implementation accepts the following command line arguments:

<!-- - `<clusters>` (required): Number of clusters for fuzzy c-means clustering -->
- `[video_path]` (required): Path to the input video file

Example usage:
```bash
python opticalflow.py 5
python opticalflow.py 3 path/to/video.mp4
```

### Funções principais

O arquivo `opticalflow.py` separa a lógica em funções pequenas para facilitar a manutenção:

- `exclude_invalid_points(...)`: analisa o histórico recente de cada ponto rastreado e remove pontos com movimento inconsistente, como mudanças bruscas de direção ou variações muito grandes de velocidade. Quando um ponto é considerado inválido, a região ao redor dele é bloqueada na máscara de detecção para evitar que o rastreador volte a capturar aquele ponto problemático imediatamente.
- `cluster_points(...)`: monta os atributos usados no agrupamento dos pontos, combinando posição atual, velocidade atual, velocidade histórica e variações do movimento. Em seguida, aplica fuzzy c-means para separar os pontos em clusters, identifica outliers com baixa pertinência ao grupo e preserva a correspondência dos IDs dos clusters entre quadros consecutivos.

Essas duas etapas trabalham juntas: a primeira limpa o conjunto de pontos rastreados e a segunda organiza os pontos válidos em grupos coerentes para visualização e análise.
