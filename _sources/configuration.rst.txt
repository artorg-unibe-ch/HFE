Configuration
=============

This section provides the configuration settings for the spline-mesher project.

HFE
---

### Mesher

- **meshing**: Type of meshing. Options are `'spline'` or `'full-block'`. Default is `'spline'`.
- **element_size**: Size of the elements. Default is `1.2747`.
- **air_elements**: Whether to keep air elements for a full-block mesh in x-y direction. Default is `False`.

### Image Processing

- **origaim_separate**: For Hosseini Dataset, whether image parameters are read from original aim. Default is `False`.
- **mask_separate**: Whether to use two separate mask files (CORTMASK and TRABMASK). Default is `True`.
- **imtype**: Type of image. Options are `'NATIVE'` or `'BMD'`. Default is `'NATIVE'`.
- **bvtv_scaling**: Scaling of BVTV. Options are `0` (no scaling) or `1` (scaling of BVTV 61um to BVTV 11.4um). Default is `1`.
- **bvtv_slope**: BVTV slope. Default is `0.963`.
- **bvtv_intercept**: BVTV intercept. Default is `0.03814`.
- **BVTVd_comparison**: Whether to run the comparison in `imutils.compute_bvtv_d_seg()`. Default is `False`.
- **BVTVd_as_BVTV**: Whether to use BVTVd as BVTV. Default is `False`.
- **SEG_correction**: Whether to apply a seg correction if BVTVd_as_BVTV is `False`. Default is `True`.
- **BMC_conservation**: Whether to conserve BMC. Default is `False`.

### Homogenization

- **fabric_type**: Type of fabric. Options are `'local'` or `'global'`. Default is `'local'`.
- **roi_bvtv_size**: Size of ROI BVTV. Default is `5`.
- **STL_tolerance**: Tolerance for STL. Default is `0.2`.
- **ROI_kernel_size_cort**: Kernel size for cortical ROI. Default is `5`.
- **ROI_kernel_size_trab**: Kernel size for trabecular ROI. Default is `5`.
- **ROI_BVTV_size_cort**: Diameter of sphere with same volume as FE element. Default is `1.3453`.
- **ROI_BVTV_size_trab**: Diameter of Arias Moreno et al. 2019. Default is `4.0`.
- **isotropic_cortex**: Whether cortex fabric is isotropic. Default is `False`.
- **orthotropic_cortex**: Whether cortex fabric is orthotropic. Default is `True`.

### Loadcase

- **full_nonlinear_loadcases**: Whether to compute non-linear loadcases. Default is `False`.
- **BC_mode**: Boundary condition mode. Default is `0`.
- **control**: Loading boundary conditions. Options are `'force'` or `'displacement'`. Default is `'displacement'`.
- **start_step_size**: Starting step size. Default is `0.2`.
- **time_for_displacement**: Time for displacement. Default is `1`.
- **min_step_size**: Minimum step size. Default is `0.0000000001`.
- **max_step_size**: Maximum step size. Default is `0.3`.
- **load_displacement**: Load displacement in mm. Default is `-1.0`.

### Abaqus

- **nlgeom**: Affects NLGEOM parameter in Abaqus simulation. Options are `'on'` or `'off'`. Default is `'on'`.
- **abaqus_nprocs**: Number of processors for Abaqus. Default is `8`.
- **abaqus_memory**: Memory for Abaqus in MB. Default is `6000`.
- **delete_odb**: Whether to delete odb after retrieving data. Default is `False`.
- **max_increments**: Maximum number of increments. Default is `1000`.
- **umat**: Path to UMAT file. Default is `02_CODE/abq/UMAT_BIPHASIC.f`.

### Strain Localisation

- **strain_analysis**: Whether to use strain localisation for the analysis. Default is `True`.

### Registration

- **registration**: Whether to use registered data for longitudinal studies. Default is `False`.

### Optimization

- **fz_max_factor**: Maximum factor for Fz. Default is `0.5`.
- **fx_fy_max_factor**: Maximum factor for Fx and Fy. Default is `1.3`.
- **mx_my_max_factor**: Maximum factor for Mx and My. Default is `1`.
- **mz_max_factor**: Maximum factor for Mz. Default is `0.8`.

### Old Config

- **nphases**: Number of phases. Default is `1`.
- **ftype**: File type. Default is `iso`.
- **verification_file**: Verification file. Default is `1`.
- **all_mask**: Whether all elements containing part of a mask are converted to FE mesh. Default is `True`.
- **adjust_element_size**: Whether to adjust element size to fit into common region. Default is `True`.

Mesh
----

### Image Settings

- **img_basepath**: Base path for the images. Default is `01_DATA`.
- **meshpath**: Path where the mesh files are stored. Default is `03_MESH`.
- **outputpath**: Path where the output files are stored. Default is `03_OUTPUT/MESHES/`.

### Meshing Settings

- **aspect**: Aspect ratio of the plots. Default is `100`.
- **_slice**: Slice of the image to be plotted. Default is `1`.
- **undersampling**: Undersampling factor of the image. Default is `1`.
- **slicing_coefficient**: Using every nth slice of the image for the spline reconstruction. Default is `20`.
- **inside_val**: Threshold value for the inside of the mask. Default is `0`.
- **outside_val**: Threshold value for the outside of the mask. Default is `1`.
- **lower_thresh**: Lower threshold for the mask. Default is `0`.
- **upper_thresh**: Upper threshold for the mask. Default is `0.9`.
- **s**: Smoothing factor of the spline. Default is `500`.
- **k**: Degree of the spline. Default is `3`.
- **interp_points**: Number of points to interpolate the spline. Default is `200`.
- **dp_simplification_outer**: Ramer-Douglas-Peucker simplification factor for the periosteal contour. Default is `3`.
- **dp_simplification_inner**: Ramer-Douglas-Peucker simplification factor for the endosteal contour. Default is `5`.
- **thickness_tol**: Minimum cortical thickness tolerance. Default is `0.5`.
- **phases**: Number of phases. Default is `2`.
- **center_square_length_factor**: Size ratio of the refinement square. Default is `0.4`.
- **mesh_order**: Order of the mesh. Options are `1` (linear) or `2` (quadratic). Default is `1`.
- **sweep_factor**: Factor for the sweep used in hydra for the sensitivity analysis. Default is `1`.
- **n_elms_longitudinal**: Number of elements longitudinally. Default is `3`.
- **n_elms_transverse_trab**: Number of transverse elements in trabecular bone. Default is `15`.
- **n_elms_transverse_cort**: Number of transverse elements in cortical bone. Default is `3`.
- **n_elms_radial**: Number of radial elements. Default is `20`.
- **ellipsoid_fitting**: Whether to use ellipsoid fitting. Default is `True`.
- **show_plots**: Whether to show plots during construction. Default is `False`.
- **show_gmsh**: Whether to show GMSH GUI. Default is `False`.
- **write_mesh**: Whether to write mesh to file. Default is `True`.
- **trab_refinement**: Whether to refine trabecular mesh at the center. Default is `False`.
- **mesh_analysis**: Whether to perform mesh analysis. Default is `True`.

Paths
-----

### Paths

- **origaimdir**: Directory for original aim files. Default is `00_ORIGAIM/TIBIA`.
- **aimdir**: Directory for aim files. Default is `01_DATA/TIBIA/`.
- **feadir**: Directory for FEA simulations. Default is `04_SIMULATIONS/TIBIA`.
- **sumdir**: Directory for summaries. Default is `05_SUMMARIES/TIBIA`.
- **commondir**: Directory for common region images. Default is `01_DATA/BATCH/FEA_noReg/IMAGES/`.
- **folder_bc_psl_loadcases**: Directory for BC PSL loadcases. Default is `02_CODE/abq/BC_PSL/`.
- **boundary_conditions**: Path to boundary conditions file. Default is `02_CODE/abq/BC_PSL/boundary_conditions.inp`.
- **odb_OF_python_script**: Path to ODB OF Python script. Default is `02_CODE/src/hfe_abq/readODB_acc.py`.
- **odb_python_script**: Path to ODB Python script. Default is `02_CODE/src/hfe_abq/readODB_acc.py`.

### Filenames

- **filename_postfix_cort_mask**: Postfix for cortical mask filename. Default is `_CORT_MASK_UNCOMP.AIM`.
- **filename_postfix_trab_mask**: Postfix for trabecular mask filename. Default is `_TRAB_MASK_UNCOMP.AIM`.
- **filename_postfix_mask**: Postfix for mask filename if mask separate is `False`. Default is `_MASK.AIM`.
- **filename_postfix_bmd**: Postfix for BMD filename. Default is `_UNCOMP.AIM`.
- **filename_postfix_seg**: Postfix for segmentation filename. Default is `_UNCOMP_SEG.AIM`.
- **filename_postfix_common**: Postfix for common region mask filename if registration is `True`. Default is `_common_region_MASK.mhd`.
- **filename_postfix_transform**: Postfix for transformation filename if registration is `True`. Default is `_transformation.tfm`.

### Version

- **verification_files**: Number of verification files. Default is `1`.
- **current_version**: Current version of the configuration. Default is `04_strain`.
- **site_bone**: Site bone. Options are `'Radius'` or `'Tibia'`. Default is `'Tibia'`.

Simulations
-----------

### Simulations

- **grayscale_filenames**: List of grayscale filenames. Example: `C0001234`.
- **folder_id**: Mapping of folder IDs. Example: `C0001234: pat_001`.