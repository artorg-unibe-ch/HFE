# Recovery of Image properties within spline hex mesh
# @Author:  Simone Poncioni, MSB
# @Date:    18.06.2024

from pathlib import Path

import gmsh
import numpy as np
import pyvista as pv
import SimpleITK as sitk


# * Cortical volume of mask vs cortical volume of mesh
# ---
def calculate_mask_volume(mhd_path: Path):
    def volume(mask_image):
        space = mask_image.GetSpacing()  # image spacing
        voxel = np.prod(space)  # voxel volume
        img = sitk.GetArrayFromImage(mask_image)
        vol = voxel * np.count_nonzero(img)
        return vol

    imsitk = sitk.ReadImage(str(mhd_path))
    vol = volume(imsitk)
    return vol


def calculate_mesh_volume(mesh_path: Path):
    gmsh.initialize()
    gmsh.open(str(mesh_path))

    gmsh.plugin.run("NewView")
    gmsh.plugin.setNumber("MeshVolume", "Dimension", 3)
    gmsh.plugin.setNumber(
        "MeshVolume", "PhysicalGroup", 1
    )  # Physical group 1 is the cortical compartment
    gmsh.plugin.run("MeshVolume")

    views = gmsh.view.getTags()
    _, _, data = gmsh.view.getListData(views[-1])
    volume = data[0][-1]

    gmsh.clear()
    gmsh.finalize()
    return volume


def difference_volume_percent(mhd_path: Path, mesh_path: Path):
    mask_vol = calculate_mask_volume(mhd_path)
    mesh_vol = calculate_mesh_volume(mesh_path)
    diff = (mesh_vol - mask_vol) / mask_vol * 100
    return diff


# * Trabecular bone volume of mask vs trabecular bone volume of mesh
def calculate_trabecular_bone_volume(vtu_path: Path):
    def get_trab_compartment(vtu):
        vtu.set_active_scalars("SDV_PBVT_Centroid")
        trab_compartment = vtu.threshold(value=0.01)
        # trab_compartment.plot(
        #     scalars="SDV_PBVT_Centroid", show_edges=True, show_grid=True, cmap="reds"
        # )
        return trab_compartment

    def calculate_cell_volume(vtu):
        sized = vtu.compute_cell_sizes()
        cell_volumes = sized.cell_data["Volume"]
        return cell_volumes

    def get_cell_bvtv(vtu):
        bvtv = vtu.cell_data["SDV_BVTVT_Centroid"]
        return bvtv

    def calculate_bv(cell_volume, bvtv):
        bv_cell = cell_volume * bvtv
        bv = np.sum(bv_cell)
        return bv

    vtu = pv.read(vtu_path)
    vtu_trab = get_trab_compartment(vtu)
    trab_cell_volumes = calculate_cell_volume(vtu_trab)
    trab_bvtv = get_cell_bvtv(vtu_trab)
    trab_bv = calculate_bv(trab_cell_volumes, trab_bvtv)
    return trab_bv


# vtu_path = Path(
#     "/home/simoneponcioni/Desktop/REPRO_CORRECTED/tests/Step-Compression_4.vtu"
# )
# trab_bv = calculate_trabecular_bone_volume(vtu_path)
# print(f"Trabecular bone volume is {trab_bv:.3f} mm^3")


# * Cortical bone volume of mask vs cortical bone volume of mesh
def calculate_cortical_bone_volume(vtu_path: Path):
    def get_cort_compartment(vtu):
        vtu.set_active_scalars("SDV_PBVT_Centroid")
        cort_compartment = vtu.threshold(value=0.001, invert=True)
        # cort_compartment.plot(
        #     scalars="SDV_PBVT_Centroid", show_edges=True, show_grid=True, cmap="reds"
        # )
        return cort_compartment

    def calculate_cell_volume(vtu):
        sized = vtu.compute_cell_sizes()
        cell_volumes = sized.cell_data["Volume"]
        return cell_volumes

    def get_cell_bvtv(vtu):
        bvtv = vtu.cell_data["SDV_BVTVC_Centroid"]
        return bvtv

    def calculate_bv(cell_volume, bvtv):
        bv_cell = cell_volume * bvtv
        bv = np.sum(bv_cell)
        return bv

    vtu = pv.read(vtu_path)
    vtu_trab = get_cort_compartment(vtu)
    trab_cell_volumes = calculate_cell_volume(vtu_trab)
    trab_bvtv = get_cell_bvtv(vtu_trab)
    trab_bv = calculate_bv(trab_cell_volumes, trab_bvtv)
    return trab_bv


def main():
    simdir = Path("04_SIMULATIONS/REPRO/IMAGES")
    # for each subdir, get the vtu with suffix '.vtu'
    res_dict = {}
    for vtu in simdir.rglob("*_with_data.vtu"):
        print(f"Processing {vtu}")
        parent_dir = str(vtu.parent).split("/")[-1].split("_")[0]
        # print(f'Parent directory:\t{parent_dir}')
        tb_bv = calculate_trabecular_bone_volume(vtu)
        ct_bv = calculate_cortical_bone_volume(vtu)
        res_dict[parent_dir] = tb_bv, ct_bv

    with open("05_SUMMARIES/REPRO/mesh_bvtv_ctbv.csv", "w") as f:
        for key in res_dict.keys():
            f.write("%s, %s\n" % (key, res_dict[key]))
    print("Done.")


if __name__ == "__main__":
    main()
