# Script that runs pipeline_accurate_latest.py
# update: pipeline_accurate_latest.py takes the grayscale filename as a parsed argument
# this code creates a list of executables to run in parallel


import logging
import warnings
from enum import Enum
from pprint import pprint
from time import time
import json

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

        time_record, summary_path = pipeline_hfe(cfg, folder_id, grayscale_filename)
        time_dict.update({grayscale_filename: time_record})
        logger.info(f"Simulation successful for {grayscale_filename}")
        results_summary.update({grayscale_filename: "Success"})
    except Exception as exc:
        # except Warning as e:
        time_dict.update({grayscale_filename: "-"})
        logger.error(f"Generated an exception: {exc}")
        logger.error(f"Simulation failed for {grayscale_filename}")
        results_summary.update({grayscale_filename: f"Failed: {exc}"})

    end_full = time()
    time_record_full = end_full - start_full
    logger.info("Execution time:")
    pprint(time_record_full, width=1)
    with open("results-summary.json", "w") as fp:
        json.dump(results_summary, fp)

    # io_utils.log_append_processingtime(summary_path, time_record_full)


@hydra.main(config_path="../cfg/", config_name="hfe_mesh_sensitivity_analysis", version_base=None)
def main(cfg: HFEConfig):
    EXECUTION_TYPE = ExecutionType.PYTHON

    if EXECUTION_TYPE == ExecutionType.PYTHON:
        standalone_execution_sequential(cfg)
    elif EXECUTION_TYPE == ExecutionType.SHELL:
        raise NotImplementedError("Shell execution is not implemented")


if __name__ == "__main__":
    main()
