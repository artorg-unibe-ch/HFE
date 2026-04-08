---
title: "Ship Python/Abaqus applications to HPC"
author: Simone Poncioni
date: 29 May 2024
geometry: margin=2cm
output: pdf_document
header-includes:
  - \usepackage{listings}
  - \lstset{
      basicstyle=\ttfamily,
      breaklines=true
    }
---

When developing research software, reproducibility and portability are critical. This guide shows how to build Docker images for HFE, publish them to Docker Hub, and run them on Ubelix through Apptainer.

## 1. Build a container with your Python application

The repository already includes two Dockerfiles in `02_CODE/docker_apptainer_hpc`: one for `gcc` and one for `ifort`. A Dockerfile defines all commands required to assemble an image. In this case:

**BASE IMAGE SETTINGS**
```Dockerfile
FROM ubuntu:24.04                       # select a compatible base image

ENV DEBIAN_FRONTEND noninteractive      # set environment variables (non-interactive means no prompts during installation)
ENV LC_ALL=en_US.UTF-8                  # ! set these locales to avoid issues with Ubelix/Rocky Linux 9
ENV LANG=en_US.UTF-8
```

**IFORT COMPILER INSTALLATION**
```Dockerfile
# install ifort compiler
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
RUN sudo apt update && sudo apt upgrade -y
RUN sudo apt-get install -y intel-basekit intel-hpckit
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh"
RUN unset PYTHONPATH
```

And so on. Follow basic recommendations [here](https://doi.org/10.1371/journal.pcbi.1008316) and [here](https://docs.docker.com/guides/) to build an image suitable for your needs. To build one of the provided images:

```bash
cd 02_CODE/docker_apptainer_hpc
docker build -f Dockerfile.ubuntu24.04.gcc -t simoneponcioni/hfe_development_gcc:latest .
docker build -f Dockerfile.ubuntu24.04.ifort -t simoneponcioni/hfe_development_ifort:latest .
```
You can also add build arguments to the Dockerfile. For example, the username and the token to a private repository on Docker Hub:

```Dockerfile
cd 02_CODE/docker_apptainer_hpc
docker build -f Dockerfile.ubuntu24.04.ifort --build-arg GITHUB_USERNAME=<username> -t <username>/<image>:<tag> .
```

You can create an account on Docker Hub and push the image to your repository. This way, you can share the image with collaborators or use it on different machines. To push the image to Docker Hub, follow these steps:

1. Log in to Docker Hub:

```bash
docker login
```

2. Tag the image with your Docker Hub username and the repository name:

```bash
docker tag tag/release-version <username>/<repository-name:tag/release-version>
```

3. Push the image to Docker Hub:

```bash
docker push <username>/<repository-name:tag/release-version>
```

Amazing! Now you have built your container with all dependencies and the code. You can share it with collaborators or use it on different machines. In the next section, I will show how to build an Apptainer image for Ubelix.

## 2. Build an Apptainer image

Building the image is relatively straightforward once you have pushed an image to Docker Hub. Log in to Ubelix and open a bash session on a node with:

```bash
<submit-node> srun --time=01:00:00 --mem-per-cpu=10G --pty bash
```

Then, follow these steps:

### Interactive build
```bash
<node> mkdir -p <path/to/apptainer>
<node> cd <path/to/apptainer>
export APPTAINER_BINDPATH="$HOME/:$HOME/" # more details on bind directories below
<node> apptainer build --force hfe_development_ifort.sif docker://simoneponcioni/hfe_development_ifort:latest
```

### Bind directories

By default, the started application runs within the container. The container works like a separate machine with its own operating system. Therefore, by default, you have no access to files and directories outside the container. This can be changed using bind paths.

If files are needed outside the container, e.g. in your HOME you can add the path to APPTAINER_BINDPATH="src1[:dest1],src2[:dest2]. All subdirectories and files will be accessible. Thus you could bind your HOME directory as:

```bash
export APPTAINER_BINDPATH="$HOME/:$HOME/"   
# or simply 
export APPTAINER_BINDPATH="$HOME"
```

### Automated build

Great! Typically, one hour should be enough to build the image. If you do not want to build it interactively, you can submit a job with the following script:

```bash
# Job name
#SBATCH --job-name="build_container"

# Runtime and memory
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=64G

# Workdir
#SBATCH --chdir=/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/apptainer
#SBATCH --out=%x.out
#SBATCH --err=%x.err


# Run command
# '--force' flag is used to overwrite the image if it already exists
srun apptainer build --force hfe_development_ifort.sif docker://simoneponcioni/hfe_development_ifort:latest
```

Further documentation can be found [here](https://hpc-unibe-ch.github.io/software/apptainer.html) and [here](https://apptainer.org/documentation/).

## 3. Run the container
Our use case is to run a Python application within the container, but which can communicate with the host machine (e.g. to read and write files, or run Abaqus). To do so, remember to bind the directories as shown above. Following this step, the container can be run with the following command (or with a job script):

### Run the container interactively
```bash
# Run command
srun apptainer exec /storage/workspaces/artorg_msb/hpc_abaqus/poncioni/apptainer/hfe_development_ifort.sif \
/bin/bash -c "source /opt/intel/oneapi/setvars.sh && source /opt/miniconda/etc/profile.d/conda.sh && conda activate hfe-essentials && python abq_submit.py"
```

Where:
```bash
srun apptainer exec # runs the container

/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/apptainer/hfe_development_ifort.sif # path to the container

/bin/bash -c # opens a bash session within the container
source /opt/intel/oneapi/setvars.sh # sources the ifort compiler settings
&& # ...within the same bash session...
source /opt/miniconda/etc/profile.d/conda.sh && conda activate hfe-essentials # sources the conda environment and activates it
&& # ...within the same bash session...
python abq_submit.py # runs the Python application
```

### Run the container with a job script

Alternatively, you can run the container with a job script. This way, you can submit the job to the Ubelix HPC cluster and run the container in the background. Here is an example of a job script:

```bash
#!/bin/bash

# User info
#SBATCH --mail-user=user.name@unibe.ch
#SBATCH --mail-type=begin,end,fail

# Job name
#SBATCH --job-name="job_name"

# Runtime and memory
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=64G

# Set Workdir
#SBATCH --chdir=/storage/workspaces/artorg_msb/hpc_abaqus/path/to/your/application
#SBATCH --output=out/hfe_%A_%a.out
#SBATCH --error=out/hfe_%A_%a.err


##################################################################################
### Load modules and Workspace

HPC_WORKSPACE=hpc_abaqus module load Workspace

unset SLURM_GTIDS

# Run command
srun apptainer exec /storage/workspaces/artorg_msb/hpc_abaqus/poncioni/apptainer/hfe_development_ifort.sif \
/bin/bash -c "source /opt/intel/oneapi/setvars.sh && source /opt/miniconda/etc/profile.d/conda.sh && conda activate hfe-essentials && python abq_submit.py"
```

Good job! You have built a container with all dependencies and code, pushed it to Docker Hub, and built an Apptainer image for Ubelix. You can now run the container interactively or with a job script while preserving reproducibility and portability.

# Taking it one step further with `GitHub Actions`

Most of the time these steps are iterative and incremental. It is important to keep track of changes and decisions using Git. GitHub provides workflows for automatic image builds and deployment.

Imagine this scenario: You are building an application that uses a newly developed Python pipeline/package. You have built a container with all the dependencies and the code, and shipped it to Docker Hub. You have also built an Apptainer image for Ubelix. Two weeks later, your colleague asks you to add a new feature to the application. You have to add the new feature to the code, rebuild the container, and ship it to Docker Hub. Do you really want to repeat these steps manually every time? No way. Let's use GitHub to automate the process:

1. Access your GitHub repository on the web:
2. Click on the "Actions" tab:
3. Click on "New workflow":
4. 'Categories/Continuous integration' -> 'Docker image'

Modify the workflow according to your needs, starting from the following template and [this documentation](https://docs.docker.com/build/ci/github-actions/):


```yaml
name: Build Docker Container

on:
  push:
    branches:
      - main
      - master 

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      -
        name: Set up QEMU
        uses: docker/setup-qemu-action@v2
      -
        name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      -
        name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      -
        name: Build and push
        uses: docker/build-push-action@v5
        with:
            context: "{{defaultContext}}:02_CODE"
            file: <Dockerfile.filename>
            push: true
            tags: ${{ secrets.DOCKERHUB_USERNAME }}/<imagename:tag>
```

Now, the next time you push your code, GitHub will build the container and push it to Docker Hub automatically. If you are working on Ubelix, remember to rebuild the Apptainer image with the latest update.

### Example of a Dockerfile for HFE simulations on Ubelix

```Dockerfile
FROM ubuntu:24.04

# Set environment variables
ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install dependencies
RUN apt-get update && apt-get install -y \
apt-utils cargo clang clang-tidy cmake \
fonts-cmu ftp gcc gcovr \
git g++ gfortran libboost-dev libcgns-dev \
libfltk1.3-dev libfreetype6-dev libgl1-mesa-dev libgl1-mesa-dri libhdf5-dev \
libocct-data-exchange-dev libocct-foundation-dev libocct-ocaf-dev libopenblas-dev libopenmpi-dev \
libpetsc-complex-dev libxfixes-dev libxcursor-dev libxft-dev libxi-dev \
libxinerama-dev libxmu-dev libslepc-complex3.19-dev mesa-common-dev python3-pip \
swig tcl-dev tk-dev valgrind vim wget \
sudo locales \
&& apt-get clean

# Set locales
RUN locale-gen en_US.UTF-8

# install ifort compiler
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
RUN sudo apt update && sudo apt upgrade -y
RUN sudo apt-get install -y intel-basekit intel-hpckit
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh"
RUN unset PYTHONPATH

# Install miniconda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
bash ~/miniconda.sh -b -p /opt/miniconda && rm ~/miniconda.sh
ENV PATH="/opt/miniconda/bin:$PATH"

# Create a conda environment
RUN conda create -n hfe-essentials python=3.12
SHELL ["conda", "run", "-n", "hfe-essentials", "/bin/bash", "-c"]
# Copy hfe requirements.text and install requirements
COPY ./requirements.text .
RUN pip install --no-cache-dir -r requirements.text
RUN pip install -U scikit-image
RUN pip install imutils

# Set GitHub username and personal access token (I am cloning a private repository below)
ARG GITHUB_USERNAME
ARG GITHUB_TOKEN
ARG USER_UID=1000
ARG USER_GID=1000

# Install the meshing package
RUN git clone https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/artorg-unibe-ch/spline_mesher.git ./pyhexspline/spline_mesher
WORKDIR ./pyhexspline/spline_mesher
RUN pip install -e .

# Change ownership of the conda environment to the hfe user
RUN useradd -ms /bin/bash hfe
RUN chown -R hfe:hfe /opt/miniconda/envs/hfe-essentials

USER hfe
WORKDIR /home/hfe
RUN mkdir -p ~/.ssh
RUN chmod 700 ~/.ssh
ENV PATH="/opt/cargo/bin:${PATH}"

# conda init, source .bashrc, conda activate
# https://pythonspeed.com/articles/activate-conda-dockerfile/
RUN conda init bash
RUN echo "conda activate hfe-essentials" >> ~/.bashrc
```
