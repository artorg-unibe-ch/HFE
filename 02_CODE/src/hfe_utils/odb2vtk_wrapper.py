import os
from pathlib import Path
import json
import logging

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.propagate = False


# flake8: noqa: E501


class Odb2VtkWrapper:
    def __init__(self, odb2vtk_path: Path, odb_path: Path, abq_path: Path, only_last_frame=False):
        self.odb2vtk_path = odb2vtk_path
        self.odb_path = Path(odb_path)
        self.vtu_path = self.odb_path.with_suffix(".vtu")
        self.abq_path = abq_path
        self.only_last_frame = only_last_frame

    def get_json(self):
        def json2dict(json_path):
            with open(json_path, "r") as f:
                return json.load(f)

        os.system(
            f"{self.abq_path} python {self.odb2vtk_path} --header 1 --odbFile {self.odb_path}"
        )
        json_path = self.odb_path.with_suffix(".json")
        print(f"JSON written to {json_path}")
        return json2dict(json_path)

    def convert(self):
        json_dict = self.get_json()
        instance = json_dict["instances"][0]
        instance_str = f'"{instance}"'

        steps = json_dict["steps"]
        stepname = steps[0][0]
        frames = [int(step.split("-frame-")[-1]) for step in steps[0][1]]
        if self.only_last_frame:
            frames = [frames[-1]]
        step_cli = f"{stepname}:{','.join(map(str, frames))}"

        logger.info(stepname)
        logger.info(step_cli)

        os.system(
            f"{self.abq_path} python {self.odb2vtk_path} --header 0 --instance {instance_str} --step {step_cli} --odbFile {self.odb_path}"
        )
        
        vtu_conv_path = self.odb_path.parent / self.odb_path.stem / f"{stepname}_{frames[0]}.vtu"

        vtu_out_path = (
            self.odb_path.parent / self.odb_path.stem / f"{self.odb_path.stem}_{frames[0]}.vtu"
        )

        logger.debug(f'vtu_conv_path: {vtu_conv_path}')
        logger.debug(f'vtu_out_path: {vtu_out_path}')
        vtu_conv_path.rename(vtu_out_path)
        return vtu_out_path.resolve()


def test():
    # abq_path = "/var/DassaultSystemes/SIMULIA/Commands/abq2021hf6"
    abq_path = "/storage/workspaces/artorg_msb/hpc_abaqus/Software/DassaultSystemes/SIMULIA/Commands/abq2024"
    # odb2vtkpath = "/home/simoneponcioni/Documents/04_TOOLS/ODB2VTK/python/odb2vtk.py"
    odb2vtkpath = "/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/TOOLS/ODB2VTK/python/odb2vtk.py"
    odb_path = "/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/HFE/04_SIMULATIONS/REPRO_PAPER/IMAGES/00000148/C2/T/C0001407_06.odb"

    wrapper = Odb2VtkWrapper(odb2vtkpath, odb_path, abq_path, only_last_frame=True)
    vtk_path = wrapper.convert()
    logger.info(vtk_path)


if __name__ == "__main__":
    test()
