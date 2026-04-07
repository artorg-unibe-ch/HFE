"""
Tests for hfe_utils/io_utils.py

Covers:
- ext()
- write_timing_summary()
- hydra_update_cfg_key()
- FileConfig class (all properties for fast and accurate pipelines)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tempfile
import csv

import pytest
from omegaconf import OmegaConf

from hfe_utils.io_utils import ext, write_timing_summary, hydra_update_cfg_key, FileConfig


# ---------------------------------------------------------------------------
# Helpers — build a minimal OmegaConf config
# ---------------------------------------------------------------------------


def make_cfg(mask_separate=True, pipeline="fast"):
    """Return a minimal OmegaConf config matching the HFEConfig schema."""
    cfg = OmegaConf.create(
        {
            "version": {"current_version": "V001"},
            "simulations": {"folder_id": {"SAMPLENAME": "SAMPLEFOLDER"}},
            "paths": {
                "feadir": "/tmp/fea",
                "aimdir": "/tmp/aim",
                "origaimdir": "/tmp/orig",
                "boundary_conditions": "/tmp/bc.txt",
            },
            "filenames": {
                "filename_postfix_bmd": "_BMD.AIM",
                "filename_postfix_mask": "_MASK.AIM",
                "filename_postfix_cort_mask": "_CORTMASK.AIM",
                "filename_postfix_trab_mask": "_TRABMASK.AIM",
                "filename_postfix_seg": "_SEG.AIM",
            },
            "image_processing": {"mask_separate": mask_separate},
        }
    )
    return cfg


SAMPLE = "SAMPLENAME"


# ---------------------------------------------------------------------------
# ext()
# ---------------------------------------------------------------------------


class TestExt:
    def test_changes_extension_string(self):
        assert ext("file.dat", ".txt") == "file.txt"

    def test_changes_extension_path(self):
        p = Path("/some/dir/file.dat")
        assert ext(p, ".txt") == "/some/dir/file.txt"

    def test_multiple_dots_in_name(self):
        result = ext("file.backup.dat", ".txt")
        assert result.endswith(".txt")

    def test_no_existing_extension(self):
        # rsplit(".", 1) on "filename" returns ["filename"] — only one part —
        # so the full string becomes the stem and new_ext is appended.
        result = ext("filename", ".csv")
        assert result == "filename.csv"

    def test_dotfile_treated_as_extension(self):
        # ".gitignore".rsplit(".", 1) → ["", "gitignore"]
        # so result is "" + ".csv" = ".csv"
        result = ext(".gitignore", ".csv")
        assert result == ".csv"

    def test_empty_extension(self):
        result = ext("file.dat", "")
        assert result == "file"

    def test_path_object_returns_string(self):
        result = ext(Path("/a/b/c.dat"), ".csv")
        assert isinstance(result, str)
        assert result.endswith(".csv")


# ---------------------------------------------------------------------------
# write_timing_summary()
# ---------------------------------------------------------------------------


class TestWriteTimingSummary:
    def _make_cfg(self, sumdir):
        return OmegaConf.create(
            {
                "paths": {"sumdir": str(sumdir)},
                "version": {"current_version": "V001"},
            }
        )

    def test_creates_file(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        write_timing_summary(cfg, "S001", {"simulation": 10.5, "full": 20.0})
        expected = tmp_path / "V001_processing_time_summary.csv"
        assert expected.exists()

    def test_writes_header_on_first_call(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        write_timing_summary(cfg, "S001", {"simulation": 5.0, "full": 8.0})
        csv_path = tmp_path / "V001_processing_time_summary.csv"
        with open(csv_path) as f:
            first_line = f.readline()
        assert "sample" in first_line
        assert "simulation_time" in first_line
        assert "full_time" in first_line

    def test_appends_data_rows(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        write_timing_summary(cfg, "S001", {"simulation": 5.0, "full": 8.0})
        write_timing_summary(cfg, "S002", {"simulation": 6.0, "full": 9.0})
        csv_path = tmp_path / "V001_processing_time_summary.csv"
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        # header + 2 data rows
        assert len(rows) == 3

    def test_na_for_missing_keys(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        write_timing_summary(cfg, "S001", {})
        csv_path = tmp_path / "V001_processing_time_summary.csv"
        with open(csv_path) as f:
            content = f.read()
        assert "NA" in content

    def test_sample_name_in_row(self, tmp_path):
        cfg = self._make_cfg(tmp_path)
        write_timing_summary(cfg, "MYSAMPLE", {"simulation": 1.0, "full": 2.0})
        csv_path = tmp_path / "V001_processing_time_summary.csv"
        with open(csv_path) as f:
            content = f.read()
        assert "MYSAMPLE" in content


# ---------------------------------------------------------------------------
# hydra_update_cfg_key()
# ---------------------------------------------------------------------------


class TestHydraUpdateCfgKey:
    def test_updates_top_level_key(self):
        cfg = OmegaConf.create({"foo": "old_value"})
        hydra_update_cfg_key(cfg, "foo", "new_value")
        assert cfg.foo == "new_value"

    def test_updates_nested_key(self):
        cfg = OmegaConf.create({"paths": {"feadir": "/old/path"}})
        hydra_update_cfg_key(cfg, "paths.feadir", "/new/path")
        assert cfg.paths.feadir == "/new/path"

    def test_updates_integer_value(self):
        cfg = OmegaConf.create({"settings": {"count": 1}})
        hydra_update_cfg_key(cfg, "settings.count", 42)
        assert cfg.settings.count == 42

    def test_updates_boolean_value(self):
        cfg = OmegaConf.create({"flags": {"enabled": False}})
        hydra_update_cfg_key(cfg, "flags.enabled", True)
        assert cfg.flags.enabled is True


# ---------------------------------------------------------------------------
# FileConfig — fast pipeline
# ---------------------------------------------------------------------------


class TestFileConfigFastPipeline:
    def setup_method(self):
        self.cfg = make_cfg(mask_separate=True, pipeline="fast")
        self.fc = FileConfig(self.cfg, SAMPLE, pipeline="fast", origaim_separate=True)

    def test_sample_stored(self):
        assert self.fc.sample == SAMPLE

    def test_file_bmd(self):
        assert str(self.fc.file_bmd) == f"{SAMPLE}_BMD.AIM"

    def test_file_gray_with_separate_origaim(self):
        assert self.fc.file_gray == Path(SAMPLE).with_suffix(".AIM")

    def test_file_gray_without_separate_origaim(self):
        fc = FileConfig(self.cfg, SAMPLE, pipeline="fast", origaim_separate=False)
        assert fc.file_gray == fc.file_bmd

    def test_raw_name_contains_origaimdir(self):
        assert "/tmp/orig" in str(self.fc.raw_name)

    def test_bmd_name_contains_origaimdir(self):
        assert "/tmp/orig" in str(self.fc.bmd_name)

    def test_boundary_conditions(self):
        assert self.fc.boundary_conditions == "/tmp/bc.txt"

    def test_file_mask_fast(self):
        assert str(self.fc.file_mask) == f"{SAMPLE}_MASK.AIM"

    def test_mask_name_fast(self):
        assert "MASK" in str(self.fc.mask_name)

    def test_inp_name_contains_version(self):
        assert "V001" in str(self.fc.inp_name)

    def test_vtk_name_contains_version(self):
        assert "V001" in str(self.fc.vtk_name)

    def test_sum_name_contains_version(self):
        assert "V001" in str(self.fc.sum_name)

    def test_ver_bpv_name_contains_version(self):
        assert "V001" in str(self.fc.ver_bpv_name)

    def test_cort_mask_name_none_for_fast(self):
        assert self.fc.cort_mask_name is None

    def test_trab_mask_name_none_for_fast(self):
        assert self.fc.trab_mask_name is None

    def test_seg_name_none_for_fast(self):
        assert self.fc.seg_name is None

    def test_set_filenames_returns_dict(self):
        d = self.fc.set_filenames()
        assert isinstance(d, dict)

    def test_set_filenames_contains_required_keys_fast(self):
        d = self.fc.set_filenames()
        for key in ("FILEBMD", "FILEGRAY", "RAWname", "BMDname", "INPname", "VTKname", "SUMname"):
            assert key in d, f"Key '{key}' missing from set_filenames() output"

    def test_set_filenames_contains_mask_fast(self):
        d = self.fc.set_filenames()
        assert "FILEMASK" in d
        assert "MASKname" in d


# ---------------------------------------------------------------------------
# FileConfig — accurate pipeline (mask_separate=True)
# ---------------------------------------------------------------------------


class TestFileConfigAccuratePipeline:
    def setup_method(self):
        self.cfg = make_cfg(mask_separate=True, pipeline="accurate")
        self.fc = FileConfig(self.cfg, SAMPLE, pipeline="accurate", origaim_separate=True)

    def test_file_mask_cort(self):
        assert str(self.fc.file_mask_cort) == f"{SAMPLE}_CORTMASK.AIM"

    def test_file_mask_trab(self):
        assert str(self.fc.file_mask_trab) == f"{SAMPLE}_TRABMASK.AIM"

    def test_file_seg(self):
        assert str(self.fc.file_seg) == f"{SAMPLE}_SEG.AIM"

    def test_cort_mask_name_not_none(self):
        assert self.fc.cort_mask_name is not None
        assert "CORTMASK" in str(self.fc.cort_mask_name)

    def test_trab_mask_name_not_none(self):
        assert self.fc.trab_mask_name is not None
        assert "TRABMASK" in str(self.fc.trab_mask_name)

    def test_seg_name_not_none(self):
        assert self.fc.seg_name is not None
        assert "SEG" in str(self.fc.seg_name)

    def test_set_filenames_accurate_keys(self):
        d = self.fc.set_filenames()
        for key in ("FILEMASKCORT", "FILEMASKTRAB", "FILESEG", "CORTMASKname", "TRABMASKname", "SEGname"):
            assert key in d, f"Key '{key}' missing from set_filenames() output for accurate pipeline"

    def test_file_mask_none_when_mask_separate(self):
        # With mask_separate=True, file_mask should be None for accurate pipeline
        assert self.fc.file_mask is None

    def test_file_mask_not_none_when_mask_not_separate(self):
        cfg = make_cfg(mask_separate=False, pipeline="accurate")
        fc = FileConfig(cfg, SAMPLE, pipeline="accurate", origaim_separate=True)
        assert fc.file_mask is not None


# ---------------------------------------------------------------------------
# FileConfig — folder structure integration
# ---------------------------------------------------------------------------


class TestFileConfigFolderStructure:
    def test_feadir_contains_folder(self):
        cfg = make_cfg()
        fc = FileConfig(cfg, SAMPLE, pipeline="fast")
        assert "SAMPLEFOLDER" in str(fc.feadir)

    def test_aimdir_contains_folder(self):
        cfg = make_cfg()
        fc = FileConfig(cfg, SAMPLE, pipeline="fast")
        assert "SAMPLEFOLDER" in str(fc.aimdir)

    def test_origaimdir_contains_folder(self):
        cfg = make_cfg()
        fc = FileConfig(cfg, SAMPLE, pipeline="fast")
        assert "SAMPLEFOLDER" in str(fc.origaimdir)
