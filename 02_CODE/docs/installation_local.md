## 🔧 Installation

This project uses Conda to manage its Python dependencies. To create a Conda environment with the required dependencies, follow these steps:

1. **Clone the repository**: Run the following command to clone the latest version of the hFE pipeline. The recursive flag also clones `ODB2VTK` as a submodule.

```sh

git clone --recursive https://github.com/artorg-unibe-ch/HFE.git
```

2. **Install Conda**: If you haven't already, download and install Conda from the [official website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

3. **Create a Conda Environment**: Run the following command to create a new Conda environment.

```sh
conda create --name hfe-essentials python=3.12
conda activate hfe-essentials
pip install -r 02_CODE/requirements.txt
```

4. **Validation of the installation with pytest**:

```sh
python -m pip install pytest && python -m pytest -q 02_CODE/tests
```
