# Homogenised Finite Elements (HFE) pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13879889.svg)](https://doi.org/10.5281/zenodo.13879889)

[![Build Docker Container - gcc](https://github.com/artorg-unibe-ch/HFE/actions/workflows/build-gcc.yml/badge.svg)](https://github.com/artorg-unibe-ch/HFE/actions/workflows/build-gcc.yml)
[![Build Docker Container - ifort](https://github.com/artorg-unibe-ch/HFE/actions/workflows/build-ifort.yml/badge.svg)](https://github.com/artorg-unibe-ch/HFE/actions/workflows/build-ifort.yml)
[![Documentation](https://github.com/artorg-unibe-ch/HFE/actions/workflows/docs.yml/badge.svg)](https://github.com/artorg-unibe-ch/HFE/actions/workflows/docs.yml)
[![Run TODO to Issue](https://github.com/artorg-unibe-ch/HFE/actions/workflows/todo_to_issue.yml/badge.svg)](https://github.com/artorg-unibe-ch/HFE/actions/workflows/todo_to_issue.yml)


👷🏼 Simone Poncioni <br> 🦴 Musculoskeletal Biomechanics Group<br> 🎓 ARTORG Center for Biomedical Engineering Research, University of Bern


## 📝 Introduction

<p style='text-align: justify;'> We present a robust and efficient standalone homogenised finite element pipeline from HR-pQCT clinical imaging data. Traditional voxel-based meshing techniques often struggle to accurately represent the complex geometry of cortical bone, leading to suboptimal mechanical simulations. To address this, our method leverages a smooth representation that significantly enhances the precision of cortical shell modelling, outperforming monophasic isotropic voxel-based meshes. By generating structured meshes, our approach ensures greater efficiency, reduced memory usage, and improved comparability across different patients or longitudinal studies. The algorithm integrates advanced image processing techniques, including contour extraction, BSpline smoothing, and mesh optimization, to produce high-quality meshes that accurately capture both cortical and trabecular compartments. </p>

## 💡 Method

![Graphical Abstract](02_CODE/docs/smooth_mesh_graph_abstract_v2.jpg)


## 🔧 Installation

For local development on a workstation, follow the [Local installation guide](02_CODE/docs/installation_local.md).

For containerized and HPC-oriented workflows, see the [Docker and Apptainer guide](02_CODE/docs/build_container.md).

Additional project documentation:
- [Code and pipeline docs](https://artorg-unibe-ch.github.io/HFE/)

## Getting started

To run HFE, update the required configuration files first. Follow the step-by-step [setup guide](02_CODE/docs/setup.md).
