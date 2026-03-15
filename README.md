# Convolutional Learning on Directed Acyclic Graphs

This repository contains the research code for the paper "Convolutional Learning on Directed Acyclic Graphs" by Samuel Rey, Hamed Ajorlou, and Gonzalo Mateos. The project studies graph convolutional architectures tailored to directed acyclic graphs (DAGs), with experiments focused on diffusion estimation and source identification.

A preprint is available on arXiv: [Convolutional Learning on Directed Acyclic Graphs](https://arxiv.org/abs/2405.03056). The paper is currently pending publication.

## What This Repository Contains

The repository combines:

- the core implementation of DAG convolutional layers and models;
- utilities for generating DAGs, graph shift operators, and synthetic diffusion data;
- notebooks used for the main experiments in the paper;
- a small set of auxiliary scripts for verification and model inspection.

The main public workflow is centered on the code in `src/` and the top-level experiment notebooks.

## Repository Structure

- `src/`: core reusable implementation of architectures, models, baselines, and graph utilities.
- `data/`: processed datasets and graph assets used by the experiments in this repository.
- `diffusion_learning.ipynb`: main notebook for diffusion learning experiments.
- `source_id.ipynb`: main notebook for source identification experiments.
- `mult_source_id.ipynb`: notebook for multi-source identification experiments.
- `tools/`: auxiliary utility and verification scripts that are not part of the main workflow.
- `dec/`: preserved secondary or legacy collaborator material. It is kept in the repository for reference, but it is not required for the main public workflow.

## Setup

Create a Python environment and install the pinned dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The repository has been used primarily with notebook-based workflows, so installing the full environment from `requirements.txt` is the recommended starting point.

## How To Run The Main Experiments

The easiest way to reproduce the main experiments is to open the notebooks from the repository root:

- `diffusion_learning.ipynb` for diffusion estimation experiments;
- `source_id.ipynb` for source identification experiments;
- `mult_source_id.ipynb` for multi-source source identification experiments.

Most reusable code lives in `src/`, while the notebooks act as experiment entrypoints and analysis drivers.

## Notes On Results And Generated Files

Experiment outputs are generated locally and are not intended to be versioned as part of the public source tree. In particular:

- `results/` stores local experiment outputs and derived tables;
- `TO_BE_DELETE/` stores temporary or deprecated files kept only for local review;
- local virtual environments, caches, and notebook checkpoints are ignored through `.gitignore`.

## Citation

If you use this repository in academic work, please cite:

```bibtex
@article{rey2024convolutional,
  title={Convolutional Learning on Directed Acyclic Graphs},
  author={Rey, Samuel and Ajorlou, Hamed and Mateos, Gonzalo},
  journal={arXiv preprint arXiv:2405.03056},
  year={2024}
}
```

## License

This repository is released under the license provided in [LICENSE](LICENSE).
