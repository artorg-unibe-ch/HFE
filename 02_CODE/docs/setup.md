# Setup and execution

## Configuration setup

### hfe.yaml

Update the following groups in your HFE run configuration before launching the pipeline:

- `defaults`: Select the simulation and path presets available in `02_CODE/cfg`.
- `hydra.sweeper.params.simulations.grayscale_filenames`: Input IDs for single or batch runs.
- `mesher` and `image_processing`: Controls for mesh generation and image preprocessing.
- `homogenization`, `loadcase`, and `abaqus`: Material mapping and solver controls.
- `strain_localisation`, `registration`, `optimization`, and `old_cfg`: Optional analysis and legacy settings.

Recommended workflow:

- Start from an existing config in `02_CODE/cfg` (for example `hfe-nodaratis.yaml`).
- Adapt `defaults` first, then set your input IDs in the Hydra sweeper.
- Keep numerical defaults unless your protocol requires a sensitivity analysis.

Example template:

```yaml
defaults:
  - simulations
  - paths
  - socket
  - mesh
  - _self_

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      simulations.grayscale_filenames: C0000000

mesher:
  meshing: spline                     # or 'full-block'
  element_size: 1.2747                # if element_size changes, re-check dependent BVTV settings
  air_elements: False                 # True keeps a full-block mesh in x-y direction

image_processing:
  origaim_separate: False             # read parameters from original AIM instead of processed BMD
  mask_separate: True                 # True if CORTMASK and TRABMASK are provided as separate files
  imtype: NATIVE                      # NATIVE or BMD
  bvtv_scaling: 1                     # 0: no scaling, 1: BVTV scaling from 61um to 11.4um
  bvtv_slope: 0.963
  bvtv_intercept: 0.03814
  BVTVd_comparison: False             # runs comparison in imutils.compute_bvtv_d_seg()
  BVTVd_as_BVTV: False                # True = BVTVd as BVTV, False = SEG as BVTV
  SEG_correction: True                # applicable when BVTVd_as_BVTV is False
  BMC_conservation: False             # True/False conservation of BMC

homogenization:
  fabric_type: local                  # 'local' or 'global'
  roi_bvtv_size: 5                    # MSL fabric (5 * 1 mm edge length = 5 mm)
  STL_tolerance: 0.2
  ROI_kernel_size_cort: 5
  ROI_kernel_size_trab: 5
  ROI_BVTV_size_cort: 1.3453          # sphere-equivalent FE element diameter
  ROI_BVTV_size_trab: 4.0             # Arias Moreno et al. 2019
  isotropic_cortex: False             # isotropic cortex fabric
  orthotropic_cortex: True            # orthotropic cortex fabric
  msl_xct_to_mil_uct: 0.911           # Simon 2020
  msl_xct_to_mil_xct: 1.415           # Hosseini 2017 / Poncioni 2025

loadcase:
  full_nonlinear_loadcases: False     # True/False nonlinear loadcases [FX, FY, MX, MY, MZ]
  BC_mode: 0                          # 0: all DOF fixed / 2: two in-plane fixed / 5: all DOF free
  control: displacement               # force or displacement
  start_step_size: 0.2
  time_for_displacement: 1
  min_step_size: 0.0000000001
  max_step_size: 0.3
  load_displacement: -0.1

abaqus:
  nlgeom: on                          # on/off
  abaqus_nprocs: 8
  abaqus_memory: 6000                 # MB
  delete_odb: False
  max_increments: 1000
  umat: 02_CODE/abq/UMAT_BIPHASIC.f   # path to UMAT file, relative to the project root

strain_localisation:
  strain_analysis: True

registration:
  registration: False

optimization:
  fz_max_factor: 0.5
  fx_fy_max_factor: 1.3
  mx_my_max_factor: 1
  mz_max_factor: 0.8

old_cfg:
  nphases: 1
  ftype: iso
  verification_file: 1
  all_mask: True                      # convert all elements intersecting mask; low-BVTV elements set to 1%
  adjust_element_size: True           # adapt element size to fit common region
```

### socket.yaml

Update the following fields in your socket configuration before running the pipeline:

- `site`: Use `local` or `remote`. This field is kept for configuration compatibility; it does not change behavior anymore.
- `abaqus`: Absolute path to your Abaqus solver executable.
- `workdir`: Base directory where your HFE project pipeline is located.
- `scratchdir`: Directory used for Abaqus scratch files. Create this directory before running jobs.
- `odb2vtk`: Absolute path to `odb2vtk.py`. Download ODB2VTK from [Arris-Composites/ODB2VTK](https://github.com/Arris-Composites/ODB2VTK) and point this field to the Python file in your local clone.

Example template:

```yaml
solver:
  site: remote # local or remote (kept for compatibility)
  abaqus: /path/to/abaqus/Commands/abq2024

socket_paths: # paths that are socket specific
  workdir: /path/to/HFE
  scratchdir: /path/to/scratch/.SCRATCH
  odb2vtk: /path/to/ODB2VTK/python/odb2vtk.py
```

### mesh.yaml

Update the following groups in your mesh configuration before running the meshing pipeline:

- `img_settings.img_basepath`: Relative or absolute path to the input image directory.
- `img_settings.meshpath`: Directory where intermediate mesh assets are written.
- `img_settings.outputpath`: Directory where final mesh outputs are exported.
- `meshing_settings`: Main numerical controls for contour extraction, smoothing, and mesh density.

Recommended workflow:

- Start with the default `meshing_settings` values.
- Change only path fields first (`img_basepath`, `meshpath`, `outputpath`).
- Tune mesh density (`n_elms_*`) only if required by your study.

Example template:

```yaml
img_settings:
  img_basepath: /path/to/01_DATA
  meshpath: /path/to/03_MESH
  outputpath: /path/to/03_OUTPUT/MESHES/

meshing_settings:
  aspect: 100                         # aspect ratio of the plots
  _slice: 1                           # slice of the image to be plotted
  undersampling: 1                    # undersampling factor of the image
  slicing_coefficient: 20             # using every nth slice of the image for the spline reconstruction
  inside_val: 0                       # threshold value for the inside of the mask
  outside_val: 1                      # threshold value for the outside of the mask
  lower_thresh: 0                     # lower threshold for the mask
  upper_thresh: 0.9                   # upper threshold for the mask
  s: 800                              # smoothing factor of the spline
  k: 3                                # degree of the spline
  interp_points: 250                  # number of points to interpolate the spline
  dp_simplification_outer: 3          # Ramer-Douglas-Peucker simplification factor for the periosteal contour
  dp_simplification_inner: 5          # Ramer-Douglas-Peucker simplification factor for the endosteal contour

  thickness_tol: 0.5                  # minimum cortical thickness tolerance: 3 * XCTII voxel size
  phases: 2                           # 1: only external contour, 2: external and internal contour
  center_square_length_factor: 0.4    # size ratio of the refinement square: 0 < l_f < 1
  mesh_order: 1                       # set order of the mesh (1: linear, 2: quadratic)
  sweep_factor: 1                     # factor for the sweep used in hydra for the sensitivity analysis
  n_elms_longitudinal: 3              # validation: 3
  n_elms_transverse_trab: 13          # validation: 13
  n_elms_transverse_cort: 3           # validation: 3
  n_elms_radial: 60                   # validation: 15; should be 40 if trab_refinement is True
  ellipsoid_fitting: True

  show_plots: False                   # show plots during construction
  show_gmsh: False                    # show gmsh GUI
  write_mesh: True                    # write mesh to file
  trab_refinement: False              # True: refine trabecular mesh at the center
  mesh_analysis: True                 # True: perform mesh analysis (plot JAC det in GMSH GUI)
```

### paths.yaml

Update path and filename fields to match your dataset layout before running batch jobs:

- `paths.*dir`: Base folders for raw input, processed images, simulation outputs, and summaries.
- `paths.commondir`: Folder containing registration/common-region files; relevant when registration is enabled.
- `paths.folder_bc_psl_loadcases` and `paths.boundary_conditions`: Boundary-condition include files.
- `paths.odb_*_python_script`: Post-processing script paths for ODB extraction.
- `filenames.*`: Dataset-specific filename postfixes used by the pipeline.
- `version.site_bone`: Set to `Radius` or `Tibia` according to the study.

Recommended workflow:

- Keep relative paths rooted at the project directory when possible.
- Confirm every path exists before launching `MULTIRUN` jobs.
- Change filename postfixes only if your input naming convention differs from the default.

Example template:

```yaml
paths:
  origaimdir: 00_ORIGAIM/DATASET_NAME/
  aimdir: 01_DATA/DATASET_NAME/
  feadir: 04_SIMULATIONS/DATASET_NAME/
  sumdir: 05_SUMMARIES/DATASET_NAME/
  commondir: 01_DATA/BATCH/FEA_noReg/IMAGES/                        # used when registration is True
  folder_bc_psl_loadcases: 02_CODE/abq/BC_PSL/
  boundary_conditions: 02_CODE/abq/BC_PSL/boundary_conditions.inp
  odb_OF_python_script: 02_CODE/src/hfe_abq/readODB_acc.py
  odb_python_script: 02_CODE/src/hfe_abq/readODB_acc.py

filenames:
  filename_postfix_cort_mask: _CORT_MASK_UNCOMP.AIM
  filename_postfix_trab_mask: _TRAB_MASK_UNCOMP.AIM
  filename_postfix_mask: _MASK.AIM                                  # used if mask_separate is False
  filename_postfix_bmd: _UNCOMP.AIM
  filename_postfix_seg: _SEG_UNCOMP.AIM
  filename_postfix_common: _common_region_MASK.AIM                  # used when registration is True
  filename_postfix_transform: _transformation.tfm                   # used when registration is True

version:
  verification_files: 1
  current_version: 001_sim
  site_bone: Radius                                                  # Radius or Tibia, only changes abaqus ODB header now
```

### simulations.yaml

Use the dataset-specific simulation preset for batch processing. The pipeline reads the case identifiers from `00_ORIGAIM/filenames.txt`, matches them against `simulations.grayscale_filenames`, and uses `simulations.folder_id` to locate each case under `00_ORIGAIM/DATASET_NAME/`.

Update the following items before launching a batch run:

- `simulations.grayscale_filenames`: Ordered list of case IDs to process.
- `simulations.folder_id`: Mapping from each case ID to its folder name in `00_ORIGAIM/DATASET_NAME/`.

Recommended workflow:

- Verify that every `folder_id` entry matches an existing directory in `00_ORIGAIM/DATASET_NAME/`.
- Use the dataset-specific preset that matches your study, for example `simulations.yaml`.

Example template:

```yaml
simulations:
  grayscale_filenames:
    C0000001
    C0000002
    C0000003

  folder_id:
    C0000001: PATH_1
    C0000002: PATH_2
    C0000003: PATH_3
```

### Update pipeline_runner.py

To make `pipeline_runner.py` load your configuration changes, set `config_name` to the stem of the configuration file you want to use.

```python
@hydra.main(config_path="../cfg/", config_name="hfe", version_base=None)
```

## Execution

### Running single simulations

Bypass the configuration `simulations.grayscale_filenames` by adding it as an argument when running the script:

```python
python path/to/pipeline_runner.py simulations.grayscale_filenames=C0000001
```

### Running simulations in batch

1. Insert all simulation.greyscale_filenames into the otherwise empty `filenames.text`. This will bypass simulation.greyscale_filenames when running whole datasets in batch.

Example `filenames.text`:

```txt
C0000001
C0000002
C0000003
```

2. Run the shell script.

```sh
greyscale_filenames=/path/to/00_ORIGAIM/filenames.txt

###    Line <i> contains greyscale_filename for run <i>
# Get greyscale_filename                                                                                                                              
greyscale_filename=$(cat $greyscale_filenames | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')

### Zero pad the task ID to match the numbering of the input files
n=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

# Run command
python 02_CODE/src/pipeline_runner.py simulations.grayscale_filenames=$greyscale_filename"
```

A template for running simulations using SLURM scheduler can be found in `02_CODE/docker_apptainer_hpc/submit_hfe.sh`:

```sh
#!/bin/bash

# User info
#SBATCH --mail-user=user@institute.com
#SBATCH --mail-type=begin,end,fail

# Job name
#SBATCH --job-name="hfe_pipeline"

# Runtime and memory
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=100G
#SBATCH --array=1-5%5

# Workdir
#SBATCH --chdir=/path/to/HFE
#SBATCH --output=out/hfe_%A_%a.out
#SBATCH --error=out/hfe_%A_%a.err

##############################################################################################################
### Load modules
HPC_WORKSPACE=hpc_abaqus module load Workspace

unset SLURM_GTIDS

### greyscale_filenames.txt contains lines with 1 greyscale_filename per line.
greyscale_filenames=/path/to/HFE/00_ORIGAIM/filenames.txt

###    Line <i> contains greyscale_filename for run <i>
# Get greyscale_filename                                                                                                                              
greyscale_filename=$(cat $greyscale_filenames | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var {print $1}')


### Zero pad the task ID to match the numbering of the input files
n=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

# Run command
srun apptainer exec /path/to/apptainer/hfe_development_ifort.sif \
/bin/bash -c "source /opt/intel/oneapi/setvars.sh && source /opt/miniconda/etc/profile.d/conda.sh && conda activate hfe-essentials && python 02_CODE/src/pipeline_runner.py simulations.grayscale_filenames=$greyscale_filename"
```
