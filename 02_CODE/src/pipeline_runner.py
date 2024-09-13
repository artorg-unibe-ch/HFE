# Script that runs pipeline_accurate_latest.py
# update: pipeline_accurate_latest.py takes the grayscale filename as a parsed argument
# this code creates a list of executables to run in parallel


import json
import logging
import warnings
import os
from enum import Enum
from pprint import pprint
from time import time

import json
from pathlib import Path

import coloredlogs  # type: ignore
import hydra
from config import HFEConfig
from hfe_accurate.hfe_accurate import pipeline_hfe
from hydra.core.config_store import ConfigStore

# flake8: noqa: E501

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
coloredlogs.install(level=logging.INFO, logger=logger)

cs = ConfigStore.instance()
cs.store(name="hfe_config", node=HFEConfig)


class ExecutionType(Enum):
    SHELL = 1
    PYTHON = 2


with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")  # Cause all warnings to always be triggered.


def standalone_execution_sequential(cfg: HFEConfig):
    start_full = time()
    time_dict = {}
    results_summary = {}
    try:
        grayscale_filename = cfg.simulations.grayscale_filenames
        folder_id = cfg.simulations.folder_id[grayscale_filename]
        log_path = (
            Path(cfg.paths.sumdir)
            / "logs"
            / f"{grayscale_filename}_pipeline_runner.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Set up a new logger for this simulation
        sim_logger = logging.getLogger(f"{LOGGING_NAME}_{grayscale_filename}")
        sim_logger.setLevel(logging.INFO)
        sim_handler = logging.FileHandler(log_path)
        sim_handler.setLevel(logging.INFO)
        sim_handler.setFormatter(formatter)
        sim_logger.addHandler(sim_handler)
        coloredlogs.install(level=logging.INFO, logger=sim_logger)

        time_record, summary_path = pipeline_hfe(cfg, folder_id, grayscale_filename)
        time_dict.update({grayscale_filename: time_record})
        sim_logger.info(f"Simulation successful for {grayscale_filename}")
        results_summary.update({grayscale_filename: "Success"})
    except Exception as exc:
        time_dict.update({grayscale_filename: "-"})
        sim_logger.error(f"Generated an exception: {exc}")
        sim_logger.error(f"Simulation failed for {grayscale_filename}")
        results_summary.update({grayscale_filename: f"Failed: {exc}"})

    end_full = time()
    time_record_full = end_full - start_full
    sim_logger.info("Execution time:")
    pprint(time_record_full, width=1)
    with open("results-summary.json", "w") as fp:
        json.dump(results_summary, fp)

    # io_utils.log_append_processingtime(summary_path, time_record_full)



@hydra.main(config_path="../cfg/", config_name="hfe-nodaratis", version_base=None)
def main(cfg: HFEConfig):
    EXECUTION_TYPE = ExecutionType.PYTHON

    if EXECUTION_TYPE == ExecutionType.PYTHON:
        standalone_execution_sequential(cfg)
    elif EXECUTION_TYPE == ExecutionType.SHELL:
        raise NotImplementedError("Shell execution is not implemented")


if __name__ == "__main__":
    main()
