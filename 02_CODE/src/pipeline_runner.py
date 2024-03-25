# Script that runs pipeline_accurate_latest.py
# update: pipeline_accurate_latest.py takes the grayscale filename as a parsed argument
# this code creates a list of executables to run in parallel

import logging
import warnings
from enum import Enum
from pprint import pprint
from time import time

from hfe_abq.hfe_accurate import pipeline_hfe
import hydra
from config import HFEConfig
from hydra.core.config_store import ConfigStore
import coloredlogs

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


def standalone_execution_sequential(cfg):
    start_full = time()
    time_dict = {}
    for grayscale_filename in cfg.simulations.grayscale_filenames:
        try:
            folder_id = cfg.simulations.folder_id[grayscale_filename]
            time_record, summary_path = pipeline_hfe(cfg, folder_id, grayscale_filename)
            time_dict.update({grayscale_filename: time_record})
            logger.info(f"Simulation successful for {grayscale_filename}")
        except Exception as exc:
            # except Warning as e:
            time_dict.update({grayscale_filename: "-"})
            print(f"Generated an exception: {exc}")
            logger.error(f"Simulation failed for {grayscale_filename}")

    end_full = time()
    time_record_full = end_full - start_full
    print("Execution time:")
    pprint(time_record_full, width=1)
    # io_utils.log_append_processingtime(summary_path, time_record_full)


@hydra.main(config_path="../cfg", config_name="hfe", version_base=None)
def main(cfg: HFEConfig):
    EXECUTION_TYPE = ExecutionType.PYTHON

    if EXECUTION_TYPE == ExecutionType.PYTHON:
        standalone_execution_sequential(cfg)
    elif EXECUTION_TYPE == ExecutionType.SHELL:
        raise NotImplementedError("Shell execution is not implemented")


if __name__ == "__main__":
    main()
