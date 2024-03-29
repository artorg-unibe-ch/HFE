import logging
import textwrap
import time
from pathlib import Path
from typing import Literal

import numpy as np

LOGGING_NAME = "HFE-ACCURATE"
logger = logging.getLogger(LOGGING_NAME)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("./pipeline_runner.log")
handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(console_handler)

# flake8: noqa: E501


def timefunc(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        round_el_time = round(elapsed_time, 0)
        logger.info(
            f"{func.__name__} for {args[0].part_name} took {round_el_time} s to execute."
        )
        return result

    return wrapper


class AbaqusWriter:
    def __init__(
        self,
        cfg,
        mesh_dir: Path,
        model_name: str,
        nodes: dict[int, list[float]],
        elms: dict[int, list[int]],
        centroids_cort: dict[int, list[float]],
        centroids_trab: dict[int, list[float]],
        m_cort_np: np.ndarray,
        m_trab_np: np.ndarray,
        RHOc: np.ndarray,
        RHOt: np.ndarray,
        PHIc: np.ndarray,
        PHIt: np.ndarray,
        mm_cort: np.ndarray,
        mm_trab: np.ndarray,
        topnodes: dict[int, int],
        botnodes: dict[int, int],
        RP_tag: int,
        RP_coords: list[float],
        NLGEOM: str,
        STEP_INC: int = 1000,
        PARAM_FLAG: int = 2,
        DENSIFICATOR_FLAG: int = 0,
        VISCOSITY_FLAG: int = 0,
        POSTYIELD_FLAG: int = 0,
    ):
        """
        Initializes an instance of the AbaqusWriter class.

        Args:
        - config: Dictionary containing the configuration parameters for the whole pipeline.
        - mesh_dir: Path to the directory containing the mesh files.
        - model_name: a string containing the name of the model. # ! to be substituted with the name coming from the config
        - nodes: a dictionary containing the node IDs and their corresponding coordinates.
        - elms: a dictionary containing the element IDs and their corresponding node IDs.
        - centroids: a dictionary containing the element IDs and their corresponding centroids.
        - botnodes: a dictionary containing the bottom nodes of the model.
        - topnodes: a dictionary containing the top nodes of the model.
        - RP_tag: an integer representing the tag of the reference point.
        - RP_coords: a list containing the coordinates of the reference point.
        - NLGEOM: a string indicating whether or not to use non-linear geometry.
        - STEP_INC: an integer representing the step increment for the simulation.
        - PARAM_FLAG: an integer representing the parameter selection (0: TRABECULAR BONE ISOTROPIC, 1: TRABECULAR BONE TRANSVERSELY ISOTROPIC, 2: TRABECULAR BONE FABRIC-BASED ORTHOTROPIC, 3: COMPACT BONE ISOTROPIC, 4: COMPACT BONE TRANSVERSELY ISOTROPIC)
        - DENSIFICATOR_FLAG: an integer activating the densificator in the UMAT (0: off, 1: on)
        - VISCOSITY_FLAG: an integer representing the viscosity flag (0: RATE-INDEPENDENT, 1: LINEAR VISCOSITY, 2: EXPONENTIAL VISC, 3: LOGARITHMIC VISC, 4: POLYNOMIAL VISC, 5: POWER LAW VISC)
        - POSTYIELD_FLAG: an integer representing the postyield behaviour (0 - PERFECT PLASTICITY, 1: EXPONENTIAL HARDENING, 2: SIMPLE SOFTENING, 3: EXPONENTIAL SOFTENING, 4: PIECEWISE SOFTENING, 5: LINEAR HARDENING)
        """
        self.cfg = cfg
        self.model_name = model_name
        self.mesh_dir = mesh_dir
        self.nodes = nodes
        self.elms = elms
        self.centroids_cort = centroids_cort
        self.centroids_trab = centroids_trab
        self.m_cort_np = m_cort_np
        self.m_trab_np = m_trab_np
        self.RHOc = RHOc
        self.RHOt = RHOt
        self.PHIc = PHIc
        self.PHIt = PHIt
        self.mm_cort = mm_cort
        self.mm_trab = mm_trab
        self.botnodes = botnodes
        self.topnodes = topnodes
        self.part_name = f"{self.model_name}"
        self.RP_tag = RP_tag
        self.RP_coords = RP_coords
        self.STEP_INC = STEP_INC
        self.NLGEOM = NLGEOM
        self.abq_dict = {}
        self.param_flag = PARAM_FLAG
        self.densificator_flag = DENSIFICATOR_FLAG
        self.viscosity_flag = VISCOSITY_FLAG
        self.postyield_flag = POSTYIELD_FLAG

    def _write_header(self):
        """
        Writes the header section of the Abaqus input file.

        Returns:
        - A string containing the header section of the Abaqus input file.
        """
        header = f"""*Heading
    ** HFE-ACCURATE SIMULATION INPUT FILE
    ** Model name: {self.model_name} ({self.cfg.version.site_bone})
    ** Generated by Simone Poncioni, MSB
    * Preprint, echo=YES, model=NO, history=NO, contact=NO
    """
        return "".join(
            line.lstrip() for line in textwrap.dedent(header).splitlines(True)
        )

    def _write_parts(self):
        """
        Writes the parts section of the Abaqus input file.

        Returns:
        - A string containing the parts section of the Abaqus input file.
        """
        parts = "**\n** PARTS\n**\n"
        parts += f"*Part, name={self.part_name}\n"
        return parts

    def _write_nodes(self, nodes):
        """
        Writes the nodes section of the Abaqus input file.

        Args:
        - nodes: a dictionary containing the node IDs and their corresponding coordinates.

        Returns:
        - A string containing the nodes section of the Abaqus input file.
        """
        nodes_str = f"*Node\n"
        for key, value in nodes.items():
            nodes_str += f"{key}, {value[0]}, {value[1]}, {value[2]}\n"
        return nodes_str

    def _get_depvars(self, nb_depvars=26):
        """
        Writes the DEPVAR section of the Abaqus input file.

        Args:
        - nb_depvars: an integer representing the number of dependent variables.

        Returns:
        - A string containing the DEPVAR section of the Abaqus input file.
        """
        depvars = f"""*DEPVAR
        {nb_depvars}
        2, DMG, Damage
        15, BVTVc, BVTVC
        16, BVTVt, BVTVT
        17, PBVc, PBVC
        18, PBVt, PBVT
        22, OFvalue, OF
        23, evect_optim1, VOPT(1,1)
        24, evect_optim2, VOPT(2,1)
        25, evect_optim3, VOPT(3,1)
        26, eval_min, DOPT(1)"""
        return "".join(
            line.lstrip() for line in textwrap.dedent(depvars).splitlines(True)
        )

    def _write_elset(self, elms: dict[int, list[int]]):
        """
        Writes the ELSET section of the Abaqus input file.

        Args:
        - elms: a dictionary containing the element IDs and their corresponding node IDs.

        Returns:
        - A string containing the ELSET section of the Abaqus input file.
        """
        elset = "**\n** ELSETS\n**\n"
        for key, _ in elms.items():
            elset += f"*Elset, elset=Elset-{key}\n"
            elset += f"{key},\n"
        return elset

    def _set_orientation(self, orientation: np.ndarray, centroids: list[float]):
        """
        Sets the orientation of the element based on the given orientation and centroids.
        Orientation offset sets the coordinate system at the centroid of the element.

        Args:
        orientation (np.ndarray): The orientation of the element as a 3x3 matrix.
        centroids (list[float]): The centroids of the element as a list of three floats.

        Returns:
        np.ndarray: The orientation of the element with the centroids added to it.

        """
        orientation_r = orientation.flatten()
        # sum the centroid's coordinates to the orientation (twice)
        orientation_offset = orientation_r[:-3] + np.tile(centroids, 2)

        # Add the last three values of the orientation
        orientation_offset = np.append(orientation_offset, centroids)
        return orientation_offset

    def _set_orientation_new(self, orientation: np.ndarray):
        """
        Sets the orientation of the element based on the given orientation and centroids.
        Orientation offset sets the coordinate system at the centroid of the element.

        Args:
        orientation (np.ndarray): The orientation of the element as a 3x3 matrix.

        Returns:
        np.ndarray: The orientation of the element.

        """
        mm = np.array(
            [
                orientation[0][0],
                orientation[1][0],
                orientation[2][0],
                orientation[0][1],
                orientation[1][1],
                orientation[2][1],
            ]
        )

        # round to 3 decimals # TODO: test functionality (is this necessary?)
        mm = np.round(mm, 3)

        return mm.flatten()

    def _write_orientation(self, abq_dict: dict[int, list[int]]) -> str:
        """
        Writes the orientation definition for each element set in the input dictionary.

        Args:
        - elms (dict[int, list[int]]): A dictionary containing the element sets for each element.

        Returns:
        - orientation (str): A string containing the orientation definition for each element set.
        """
        orientation = "**\n** ORIENTATIONS\n**\n"

        # Loop over each element set in the input dictionary
        for key, value in abq_dict.items():
            # orientation += f"*Orientation, name=Orient-{key}, SYSTEM=RECTANGULAR, DEFINITION=COORDINATES\n"
            orientation += f"*Orientation, name=Orient-{key}\n"
            ori = self._set_orientation(value["mm"], value["centroid"])
            # ori = self._set_orientation_new(value["mm"])
            orientation += ", ".join(map(str, ori))
            orientation += "\n"
            # orientation += "1, 0\n"
            orientation += "1, 0.\n"

        return orientation

    def _write_section(self, elms: dict[int, list[int]]) -> str:
        """
        Writes the section definition for each element set in the input dictionary.

        Args:
        - elms (dict[int, list[int]]): A dictionary containing the element sets for each element.

        Returns:
        - section (str): A string containing the section definition for each element set.
        """
        section = "**\n** SECTIONS\n**\n"
        for key, _ in elms.items():
            section += f"*Solid Section, elset=Elset-{key}, material=Material-{key}, orientation=Orient-{key}\n"
            section += f",\n"
        return section

    def _write_elms(self, elms: dict[int, list[int]]) -> str:
        """
        Returns the Abaqus input file for the elements.

        Args:
            elms (dict[int, list[int]]): A dictionary containing the element IDs and their corresponding node IDs.

        Returns:
            str: The Abaqus input file for the elements.
        """
        elms_str = ""
        element_type = None
        gmsh_to_abq_sort = list(range(27))
        if len(elms[1]) == 8:
            element_type = "C3D8"  # 8-node hexahedron
        elif len(elms[1]) == 27:
            # sorting elements from gmsh connectivity to abq connectivity
            gmsh_to_abq_sort = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                11,
                13,
                9,
                16,
                18,
                19,
                17,
                10,
                12,
                14,
                15,
                26,
                20,
                25,
                21,
                23,
                24,
                22,
            ]
            element_type = "C3D27"

        elms_str += f"*Element, type={element_type}\n"
        sorted_values = []
        for key, values in elms.items():
            if element_type == "C3D27":
                sorted_values = [values[i] for i in gmsh_to_abq_sort]
            elif element_type == "C3D8":
                sorted_values = values
            values_str = ", ".join(map(str, sorted_values))
            full_str = f"{key}, {values_str}"
            elms_str += f"{full_str}\n"
        return elms_str

    def _write_material(self, abq_dict: dict):
        """
        Returns the Abaqus input file for the material properties.

        Args:
            elms (dict[int, list[int]]): A dictionary containing the element IDs and their corresponding node IDs.

        Returns:
            str: The Abaqus input file for the material properties.
        """
        umat_name = Path(self.cfg.abaqus.umat).name
        if umat_name == "UMAT_BIPHASIC.f":
            constants_number = int(7)
            constants_comment = str(
                "**BVTVcort, BVTVtrab, BPVcort, BPVtrab, eigenvalue min, eigenvalue mid, eigenvalue max"
            )
            i_nb_depvars = 26
            depvars = self._get_depvars(nb_depvars=i_nb_depvars)

        elif umat_name == "UMAT_QUADRIC_PRIMAL.f":
            constants_number = int(9)
            constants_comment = str(
                "**Parameter flag, BVTVbone, BPVbone, eigenvalue min, eigenvalue mid, eigenvalue max, densificator flag, viscosity flag, postyield flag"
            )
            i_nb_depvars = 22
        else:
            raise ValueError("UMAT not recognized")

        material = "**\n** MATERIALS\n**\n"
        for key, value in abq_dict.items():
            material_name = f"Material-{key}"
            material += f"*Material, name={material_name}\n"
            material += f"*Depvar\n{i_nb_depvars},\n"
            material += f"*User material, constants={constants_number}, UNSYMM, Type=Mechanical\n"
            material += f"{constants_comment}\n"
            m = ", ".join(map(str, value["m"].flatten()))
            if umat_name == "UMAT_BIPHASIC.f":
                constants = [
                    np.round(value["RHOc"], 5),
                    np.round(value["RHOt"], 5),
                    np.round(value["PHIc"], 5),
                    np.round(value["PHIt"], 5),
                    m,
                ]
                material += ", ".join(map(str, constants))
                material += "\n"

            elif umat_name == "UMAT_QUADRIC_PRIMAL.f":
                constants = [
                    self.param_flag,
                    value["RHO"],
                    value["PHI"],
                    m,
                    self.densificator_flag,
                    self.viscosity_flag,
                ]
                material += ", ".join(map(str, constants))
                material += "\n"
                material += (
                    str(self.postyield_flag) + "\n"
                )  # avoids having >8 params per line
            else:
                raise ValueError("UMAT not recognized")
        return material

    def _write_boundary_conditions(self, BC_DEF: list[int], BOTNODES: str):
        """
        Returns the Abaqus input file for boundary conditions.

        Args:
            BC_DEF (list[int]): A list containing the boundary condition values.
            BOTNODES (str): The name of the bottom nodes.

        Returns:
            str: The Abaqus input file for boundary conditions.
        """
        return f"**\n** BOUNDARY CONDITIONS\n**\n*Boundary, Type=Displacement\n{self.part_name}.{BOTNODES}, {BC_DEF[0]}, {BC_DEF[1]}, {BC_DEF[2]}\n"

    def _write_ref_node(self, RP_TAG: int, RP: list[float]):
        """
        Returns the Abaqus input file for a reference node.

        Args:
            RP_TAG (int): The tag of the reference point.
            RP (list[float]): The coordinates of the reference point.

        Returns:
            str: The Abaqus input file for a reference node.
        """
        return f"*Node\n{RP_TAG}, {RP[0]}, {RP[1]}, {RP[2]}\n"

    def _write_nset(self, NSET: str, NSET_DICT: dict[int, int]):
        """
        Returns the Abaqus input file for a node set.

        Args:
            NSET (str): The name of the node set.
            NSET_DICT (dict[int, int]): A dictionary containing the node IDs and their values.

        Returns:
            tuple: A tuple containing the header and values lines of the Abaqus input file for a node set.
        """
        head = f"*NSET, NSET={NSET}\n"
        values = list(NSET_DICT.keys())
        values_chunks = [values[i : i + 8] for i in range(0, len(values), 8)]
        values_lines = ""
        for chunk in values_chunks:
            values_str = ",\t".join("{:<4}".format(str(x)) for x in chunk)
            values_lines += f"{values_str}\n"
        return head, values_lines

    def _write_kin_coupling(self, NSET: str, NODES: str):
        """
        Returns the Abaqus input file for a kinematic coupling constraint.

        Args:
            NSET (str): The name of the reference node set.
            NODES (str): The name of the nodes set.

        Returns:
            str: The Abaqus input file for a kinematic coupling constraint.
        """
        constraint = "Kinematic-Coupling"
        surf_name = f"{self.part_name}-{NODES}_CNS_1"
        kin_coupl = f"""
        *Surface, type=NODE, name={surf_name}, internal
        {self.part_name}.{NODES}, 1.
        ** Constraint: {constraint}
        *Coupling, constraint name={constraint}, ref node={NSET}, surface={surf_name}
        *Kinematic
        """
        return textwrap.dedent(kin_coupl)

    def _write_step(self, STEP_INC: int, NLGEOM: str, step_name: str):
        """
        Returns the Abaqus input file for a step.

        Args:
            STEP_INC (int): The increment for the step.
            NLGEOM (str): The type of geometry for the step.
            step_name (str): The name of the step.

        Returns:
            str: The Abaqus input file for a step.
        """
        if self.cfg.abaqus.nlgeom is True:
            NLGEOM = "YES"
        elif self.cfg.abaqus.nlgeom is False:
            NLGEOM = "NO"
        else:
            raise ValueError("nlgeom must be on or off in configuration file")
        str_comment = f"** STEP: Step-{step_name}\n**\n"
        str_step = f"*Step, name=Step-{step_name}, nlgeom={NLGEOM}, inc={STEP_INC}, Amplitude=RAMP, unsymm=YES\n"
        return str_comment + str_step

    def _write_output(self):
        """
        Returns the Abaqus input file for output requests.

        Returns:
            str: The Abaqus input file for output requests.
        """
        output = f"""**
        ** OUTPUT REQUESTS
        **
        *OUTPUT, FIELD"""
        return "".join(
            line.lstrip() for line in textwrap.dedent(output).splitlines(True)
        )

    def _write_output_history(self, REF_NODE: str):
        """
        Returns the Abaqus input file for output history of a reference node.

        Args:
            REF_NODE (str): The reference node.

        Returns:
            str: The Abaqus input file for output history of a reference node.
        """
        out_hist = f"""
            *Node Output
            U,
            RF,
            CF,
            *Element Output, directions=YES
            SDV
            ** HISTORY OUTPUT
            *Output, history
            *Node Output, nset={REF_NODE}
            U,
            RF
            *NODE PRINT, NSET=REF_NODE, FREQUENCY=1, SUMMARY=NO
            U,
            RF,
            CF,
            *End Step
        """
        return textwrap.dedent(out_hist)

    def _write_boundary_conditions_rp(
        self,
        RP_TAG: str,
    ):
        """
        Returns the Abaqus input file for the boundary conditions of a Reference Point.

        Args:
            RP_TAG (str): The reference point tag.

        Returns:
            str: The Abaqus input file for the boundary conditions of a Reference Point.
        """
        magnitude = self.cfg.loadcase.load_displacement
        DISPLACEMENT = [0, 0, magnitude, 0, 0, 0]

        bc_str = textwrap.dedent(
            f"""
        *Static
        ** Initial increment size, Time period, Min increment size, Max increment size
        {self.cfg.loadcase.start_step_size}, {self.cfg.loadcase.time_for_displacement}, {self.cfg.loadcase.min_step_size}, {self.cfg.loadcase.max_step_size}
        
        **
        ** BOUNDARY CONDITIONS
        **
        *Boundary, Type=Displacement
        {RP_TAG}, 1, 1, {DISPLACEMENT[0]}
        {RP_TAG}, 2, 2, {DISPLACEMENT[1]}
        {RP_TAG}, 3, 3, {DISPLACEMENT[2]}
        {RP_TAG}, 4, 4, {DISPLACEMENT[3]}
        {RP_TAG}, 5, 5, {DISPLACEMENT[4]}
        {RP_TAG}, 6, 6, {DISPLACEMENT[5]}
        """
        )
        return bc_str

    def abq_part(self):
        """
        Returns information at the part level (Part + Instance).

        Returns:
            str: The Abaqus input file for the part.
        """
        header = self._write_header()
        parts = self._write_parts()
        nodes = self._write_nodes(nodes=self.nodes)
        elements = self._write_elms(elms=self.elms)
        BOTNODES_header, BOTNODES_SET = self._write_nset("BOTNODES", self.botnodes)
        TOPNODES_header, TOPNODES_SET = self._write_nset("TOPNODES", self.topnodes)

        part_t0 = header + parts + nodes + elements + "\n"
        gen_nset = f"*Nset, nset={self.part_name}, generate\n1, {len(self.nodes)}, 1\n"
        gen_elset_cort = f"*Elset, elset={self.part_name}_CORT, generate\n{min(self.centroids_cort)}, {max(self.centroids_cort)}, 1\n"

        gen_elset_trab = f"*Elset, elset={self.part_name}_TRAB, generate\n{min(self.centroids_trab)}, {max(self.centroids_trab)}, 1\n"

        part_t1 = gen_nset + gen_elset_cort + gen_elset_trab
        part_t2 = BOTNODES_header + BOTNODES_SET + TOPNODES_header + TOPNODES_SET

        elsets = self._write_elset(self.elms)
        sections = self._write_section(self.elms)
        orientation = self._write_orientation(self.abq_dict)

        part_t3 = elsets + sections + orientation

        part = part_t0 + part_t1 + part_t2 + part_t3
        part += "*End Part\n"
        return part

    def abq_assembly(self):
        """
        Returns information at the assembly level (Assembly + Model Instance).

        Returns:
            str: The Abaqus input file for the assembly.
        """
        assembly_statement = "**\n** ASSEMBLY\n**\n"
        assembly = "*Assembly, name=Assembly\n**\n"
        instance: Literal[
            f"*Instance, name={self.part_name}, part={self.part_name}\n"
        ] = f"*Instance, name={self.part_name}, part={self.part_name}\n"
        end_instance = "*End Instance\n**\n"

        ref_node = self._write_ref_node(self.RP_tag, self.RP_coords)

        # write dictionary of RP tag with index and tag to pass it to _write_nset
        RP_dict = {self.RP_tag: self.RP_tag}
        ref_node_set = self._write_nset("REF_NODE", RP_dict)
        ref_node_set_coupling = self._write_kin_coupling("REF_NODE", "TOPNODES")
        end_assembly = "*End Assembly\n"

        abq_assembly = (
            assembly_statement
            + assembly
            + instance
            + end_instance
            + ref_node
            + ref_node_set[0]  # header
            + ref_node_set[1]  # set
            + ref_node_set_coupling
            + end_assembly
        )
        return abq_assembly

    def abq_model(self):
        """
        Writes the Abaqus input file for the model.

        Returns:
            str: The Abaqus input file for the model.
        """
        materials = self._write_material(self.abq_dict)
        # materials = self._write_material_test(self.abq_dict)
        boundary_conditions = self._write_boundary_conditions([1, 3, 0], "BOTNODES")
        spacing = (
            "** ----------------------------------------------------------------\n**\n"
        )
        step = self._write_step(self.STEP_INC, self.NLGEOM, step_name="Compression")
        boundary_conditions_ref_point = self._write_boundary_conditions_rp(
            RP_TAG="REF_NODE"
        )

        model = (
            materials
            + boundary_conditions
            + spacing
            + step
            + boundary_conditions_ref_point
        )
        return model

    def abq_history(self):
        """
        Writes the Abaqus input file for the output history.

        Returns:
            str: The Abaqus input file for the output history.
        """
        output = self._write_output()
        output_history = self._write_output_history(REF_NODE="REF_NODE")
        history = output + output_history
        return history

    @timefunc
    def abaqus_writer(self, _ver: str):
        """
        Writes the Abaqus input file for the current part.

        Returns:
            None
        """
        inp_path = Path(f"{self.mesh_dir}/{self.part_name}_V_{_ver}.inp")

        with open(inp_path, "w") as f:
            f.write(self.abq_part())
            f.write(self.abq_assembly())
            f.write(self.abq_model())
            f.write(self.abq_history())

        logger.info(f"Input file saved in {inp_path.absolute()}")
        return inp_path

    def abq_dictionary(self, umat_name: str):
        # concatenate cort with trab
        self.m = np.concatenate((self.m_cort_np, self.m_trab_np), axis=0)
        self.mm = np.concatenate((self.mm_cort, self.mm_trab), axis=0)

        # make RHOc, RHOt, PHIc, PHIt a dictionary with element number as key
        self.RHOc = {i: self.RHOc[idx] for idx, i in enumerate(self.centroids_cort)}
        self.RHOt = {i: self.RHOt[idx] for idx, i in enumerate(self.centroids_trab)}

        self.PHIc = {i: self.PHIc[idx] for idx, i in enumerate(self.centroids_cort)}
        self.PHIt = {i: self.PHIt[idx] for idx, i in enumerate(self.centroids_trab)}

        abq_dict = {}
        # keys: element number
        # values: centroids, mm, elset, RHOc, RHOt, PHIc, PHIt
        # ! remember to change the name of the UMAT in the if statement
        if umat_name == "UMAT_BIPHASIC.f":
            for key, value in self.centroids_cort.items():
                abq_dict[key] = {
                    "centroid": value,
                    "m": self.m[key - 1],
                    "mm": self.mm[key - 1],
                    "elset": self.elms[key],
                    "RHOc": self.RHOc[key],
                    "RHOt": 0,
                    "PHIc": self.PHIc[key],
                    "PHIt": 0,
                }

            # keys: element number
            # values: centroids, mm, elset, RHOc, RHOt, PHIc, PHIt
            for key, value in self.centroids_trab.items():
                abq_dict[key] = {
                    "centroid": value,
                    "m": self.m[key - 1],
                    "mm": self.mm[key - 1],
                    "elset": self.elms[key],
                    "RHOc": 0,
                    "RHOt": self.RHOt[key],
                    "PHIc": 0,
                    "PHIt": self.PHIt[key],
                }

        elif umat_name == "UMAT_QUADRIC_PRIMAL.f":
            # merge cort and trab into single dictionary
            self.centroids = dict(self.centroids_cort)
            self.centroids.update(self.centroids_trab)
            self.RHO = dict(self.RHOc)
            self.RHO.update(self.RHOt)
            self.PHI = dict(self.PHIc)
            self.PHI.update(self.PHIt)

            # keys: element number
            # values: centroids, mm, elset, RHO, PHI
            for key, value in self.centroids.items():
                abq_dict[key] = {
                    "centroid": value,
                    "m": self.m[key - 1],
                    "mm": self.mm[key - 1],
                    "elset": self.elms[key],
                    "RHO": self.RHO[key],
                    "PHI": self.PHI[key],
                }
        else:
            raise ValueError("UMAT not recognized")

        self.abq_dict = abq_dict
        return abq_dict
