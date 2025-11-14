#!/bin/bash

# User info
#SBATCH --mail-user=simone.poncioni@unibe.ch
#SBATCH --mail-type=begin,end,fail

# Job name
#SBATCH --job-name="build_container"

# Runtime and memory
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=64G

# Workdir
#SBATCH --chdir=/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/apptainer
#SBATCH --out=%x.out
#SBATCH --err=%x.err


# Run command
srun apptainer build --force hfe_development_ifort.sif docker://simoneponcioni/hfe_development_ifort:latest

