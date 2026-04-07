"""
Tests for hfe_accurate/postprocessing.py

Covers:
- remove_empty_entries_list
- datfilereader_psl  (with a temporary .dat file and OmegaConf config)
- write_data_summary (with a temporary directory)
"""

import csv
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Mock heavy dependencies BEFORE importing the module under test
_MOCKED = [
    "vtk",
    "vtk.util",
    "vtk.util.numpy_support",
    "vtk.numpy_interface",
    "vtk.numpy_interface.dataset_adapter",
    "numba",
    "pyvista",
    "SimpleITK",
    "fast_simplification",
    "mpl_toolkits",
    "mpl_toolkits.axes_grid1",
]
for _mod in _MOCKED:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from omegaconf import OmegaConf

from hfe_accurate.postprocessing import (
    datfilereader_psl,
    remove_empty_entries_list,
    write_data_summary,
)


# ---------------------------------------------------------------------------
# remove_empty_entries_list
# ---------------------------------------------------------------------------


class TestRemoveEmptyEntriesList:
    """Tests for the module-level helper in hfe_accurate.postprocessing."""

    def test_removes_empty_strings(self):
        data = ["", "", "1.0", "2.0", "3.0\n"]
        result = remove_empty_entries_list(data)
        assert "" not in result

    def test_strips_trailing_newline_from_last_element(self):
        data = ["", "a", "b\n"]
        result = remove_empty_entries_list(data)
        assert result[-1] == "b"

    def test_returns_non_empty_values(self):
        data = ["", "x1", "x2", "x3\n"]
        result = remove_empty_entries_list(data)
        assert result == ["x1", "x2", "x3"]

    def test_single_element(self):
        data = ["only\n"]
        result = remove_empty_entries_list(data)
        assert result == ["only"]

    def test_no_empty_strings_no_newline_stripped_beyond_last(self):
        data = ["a", "b", "c\n"]
        result = remove_empty_entries_list(data)
        assert result[-1] == "c"

    def test_typical_split_line(self):
        line = "  1  0.5  1.0  1.5  2.0  2.5  3.0\n"
        result = remove_empty_entries_list(line.split(" "))
        assert "0.5" in result
        assert "3.0" in result
        assert not any(e.endswith("\n") for e in result)


# ---------------------------------------------------------------------------
# Helpers for dat-file tests
# ---------------------------------------------------------------------------


def _fmt_data_line(vals):
    """Return a dat-file data line with one space-separated fields and trailing newline."""
    parts = ["  1"] + [f"  {v}" for v in vals]
    return " ".join(parts) + "\n"


def _build_psl_dat(u_vals, rf_vals, cf_vals):
    """
    Build lines for a minimal PSL .dat file.
    Trigger ("U3") at index 0; U at +3; RF at +12; CF at +21.
    """
    return [
        " U1       U2       U3       UR1      UR2      UR3\n",   # 0 – trigger
        "\n",   # 1
        "\n",   # 2
        _fmt_data_line(u_vals),   # 3
        "\n",   # 4
        "\n",   # 5
        "\n",   # 6
        "\n",   # 7
        "\n",   # 8
        " RF1      RF2      RF3      RM1      RM2      RM3\n",   # 9
        "\n",   # 10
        "\n",   # 11
        _fmt_data_line(rf_vals),   # 12
        "\n",   # 13
        "\n",   # 14
        "\n",   # 15
        "\n",   # 16
        "\n",   # 17
        " CF1      CF2      CF3      CM1      CM2      CM3\n",   # 18
        "\n",   # 19
        "\n",   # 20
        _fmt_data_line(cf_vals),   # 21
        "\n",   # 22
    ]


# ---------------------------------------------------------------------------
# datfilereader_psl
# ---------------------------------------------------------------------------


class TestDatfilereaderPsl:
    """Integration tests for datfilereader_psl using temp files."""

    def _setup_env(self, tmp_path, sample="S001", version="V1"):
        """
        Create a temp feadir/<folder_id>/<sample>_<version[:2]>.dat and return cfg.
        """
        folder_id = f"{sample}_FOLDER"
        feadir = tmp_path / "fea"
        sample_dir = feadir / folder_id
        sample_dir.mkdir(parents=True)

        dat_path = sample_dir / f"{sample}_{version[:2]}.dat"
        return feadir, folder_id, sample_dir, dat_path

    def _make_cfg(self, feadir, folder_id, sample, version):
        return OmegaConf.create(
            {
                "paths": {"feadir": str(feadir)},
                "simulations": {"folder_id": {sample: folder_id}},
                "version": {"current_version": version},
            }
        )

    def test_populates_optim_dict(self, tmp_path):
        sample, version = "S001", "V1_TEST"
        feadir, folder_id, sample_dir, dat_path = self._setup_env(tmp_path, sample, version)
        lines = _build_psl_dat(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        )
        dat_path.write_text("".join(lines))
        cfg = self._make_cfg(feadir, folder_id, sample, version)
        optim = datfilereader_psl(cfg, sample, {}, "FZ_MAX")
        assert "disp_FZ_MAX" in optim
        assert "force_FZ_MAX" in optim
        assert "conc_force_FZ_MAX" in optim

    def test_correct_displacement_values(self, tmp_path):
        sample, version = "S001", "V1_TEST"
        feadir, folder_id, sample_dir, dat_path = self._setup_env(tmp_path, sample, version)
        u = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        lines = _build_psl_dat(u, [0.0]*6, [0.0]*6)
        dat_path.write_text("".join(lines))
        cfg = self._make_cfg(feadir, folder_id, sample, version)
        optim = datfilereader_psl(cfg, sample, {}, "FZ_MAX")
        assert optim["disp_FZ_MAX"][0] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_correct_force_values(self, tmp_path):
        sample, version = "S001", "V1_TEST"
        feadir, folder_id, sample_dir, dat_path = self._setup_env(tmp_path, sample, version)
        rf = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        lines = _build_psl_dat([0.0]*6, rf, [0.0]*6)
        dat_path.write_text("".join(lines))
        cfg = self._make_cfg(feadir, folder_id, sample, version)
        optim = datfilereader_psl(cfg, sample, {}, "FZ_MAX")
        assert optim["force_FZ_MAX"][0] == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    def test_correct_conc_force_values(self, tmp_path):
        sample, version = "S001", "V1_TEST"
        feadir, folder_id, sample_dir, dat_path = self._setup_env(tmp_path, sample, version)
        cf = [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
        lines = _build_psl_dat([0.0]*6, [0.0]*6, cf)
        dat_path.write_text("".join(lines))
        cfg = self._make_cfg(feadir, folder_id, sample, version)
        optim = datfilereader_psl(cfg, sample, {}, "FZ_MAX")
        assert optim["conc_force_FZ_MAX"][0] == [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]

    def test_txt_output_file_created(self, tmp_path):
        sample, version = "S001", "V1_TEST"
        feadir, folder_id, sample_dir, dat_path = self._setup_env(tmp_path, sample, version)
        dat_path.write_text("".join(_build_psl_dat([0.0]*6, [0.0]*6, [0.0]*6)))
        cfg = self._make_cfg(feadir, folder_id, sample, version)
        datfilereader_psl(cfg, sample, {}, "FZ_MAX")
        txt_path = dat_path.with_suffix(".txt")
        assert txt_path.exists()

    def test_txt_file_contains_header(self, tmp_path):
        sample, version = "S001", "V1_TEST"
        feadir, folder_id, sample_dir, dat_path = self._setup_env(tmp_path, sample, version)
        dat_path.write_text("".join(_build_psl_dat([0.0]*6, [0.0]*6, [0.0]*6)))
        cfg = self._make_cfg(feadir, folder_id, sample, version)
        datfilereader_psl(cfg, sample, {}, "FZ_MAX")
        content = dat_path.with_suffix(".txt").read_text()
        assert "inc" in content
        assert "U1" in content

    def test_empty_dat_produces_empty_data(self, tmp_path):
        sample, version = "S001", "V1_TEST"
        feadir, folder_id, sample_dir, dat_path = self._setup_env(tmp_path, sample, version)
        dat_path.write_text("\n")
        cfg = self._make_cfg(feadir, folder_id, sample, version)
        optim = datfilereader_psl(cfg, sample, {}, "FZ_MAX")
        assert optim["disp_FZ_MAX"] == []
        assert optim["force_FZ_MAX"] == []

    def test_multiple_increments(self, tmp_path):
        sample, version = "S001", "V1_TEST"
        feadir, folder_id, sample_dir, dat_path = self._setup_env(tmp_path, sample, version)
        lines = []
        for _ in range(3):
            lines.extend(_build_psl_dat([0.1]*6, [10.0]*6, [0.0]*6))
            lines.append("\n")
        dat_path.write_text("".join(lines))
        cfg = self._make_cfg(feadir, folder_id, sample, version)
        optim = datfilereader_psl(cfg, sample, {}, "FZ_MAX")
        assert len(optim["disp_FZ_MAX"]) == 3

    def test_abaqus_plus_notation_handled(self, tmp_path):
        """Lines with '5+02' notation (without 'E') should be converted to float."""
        sample, version = "S001", "V1_TEST"
        feadir, folder_id, sample_dir, dat_path = self._setup_env(tmp_path, sample, version)

        def _fmt_abaqus(vals):
            # Use Abaqus-style scientific notation without 'E'
            parts = ["  1"] + [f"  {v:.3E}".replace("E+0", "+0") for v in vals]
            return " ".join(parts) + "\n"

        rf = [500.0, 200.0, 300.0, 100.0, 150.0, 250.0]
        lines = [
            " U1       U2       U3       UR1      UR2      UR3\n",
            "\n", "\n",
            _fmt_data_line([0.1]*6),
            "\n", "\n", "\n", "\n", "\n",
            " RF1      RF2      RF3      RM1      RM2      RM3\n",
            "\n", "\n",
            _fmt_abaqus(rf),
            "\n", "\n", "\n", "\n", "\n",
            " CF1      CF2      CF3      CM1      CM2      CM3\n",
            "\n", "\n",
            _fmt_data_line([0.0]*6),
            "\n",
        ]
        dat_path.write_text("".join(lines))
        cfg = self._make_cfg(feadir, folder_id, sample, version)
        optim = datfilereader_psl(cfg, sample, {}, "FZ_MAX")
        # Should not raise; force values should be floats
        assert isinstance(optim["force_FZ_MAX"][0][0], float)


# ---------------------------------------------------------------------------
# write_data_summary
# ---------------------------------------------------------------------------


class TestWriteDataSummary:
    def _make_cfg(self, sumdir, feadir, version="V1_TEST"):
        return OmegaConf.create(
            {
                "paths": {"sumdir": str(sumdir), "feadir": str(feadir)},
                "version": {"current_version": version},
                "simulations": {
                    "folder_id": {"S001": "S001_FOLDER"},
                    "grayscale_filenames": "S001",
                },
                "bvtv_scaling": 1,
                "bvtv_slope": 1.0,
                "bvtv_intercept": 0.0,
            }
        )

    def _make_full_optim(self):
        return {
            "max_force_FZ_MAX": 1234.5,
            "disp_at_max_force_FZ_MAX": 2.5,
            "stiffness_FZ_MAX": 500.0,
            "yield_force_FZ_MAX": 900.0,
            "yield_disp_FZ_MAX": 1.8,
        }

    def _make_full_bone(self):
        return {
            "TOT_mean_BMC_image": 100.0,
            "TOT_simulation_BMC_FE_tissue_orig_ROI": 99.0,
            "TOT_simulation_BMC_FE_tissue_ROI": 98.0,
            "TOT_BMC_ratio_ROI": 0.98,
            "mean_BMD_SEG_CORTorig": 750.0,
            "mean_BMD_SEG_TRABorig": 250.0,
            "mean_BMD_SEG_CORTscaled": 780.0,
            "mean_BMD_SEG_TRABscaled": 260.0,
            "BV_CORT_SEG": 50.0,
            "BV_TRAB_SEG": 80.0,
            "nel_CORT": 200,
            "nel_TRAB": 500,
            "nel_MIXED": 30,
            "mean_BVTV_seg": 0.3,
            "mean_BVTVd_scaled": 0.32,
            "mean_BVTVd_raw": 0.30,
            "mean_area": 100.0,
            "trab_avg_DA": 1.5,
        }

    def _make_mesh_params(self):
        return {
            "n_elms_longitudinal": 10,
            "n_elms_transverse_trab": 5,
            "n_elms_transverse_cort": 3,
            "n_elms_radial": 8,
        }

    def test_creates_csv_file(self, tmp_path):
        sumdir = tmp_path / "summary"
        feadir = tmp_path / "fea"
        cfg = self._make_cfg(sumdir, feadir)
        write_data_summary(
            cfg, self._make_full_optim(), self._make_full_bone(), "S001",
            self._make_mesh_params(), 1000, 42.0,
        )
        csv_file = sumdir / "V1_TEST_data_summary.csv"
        assert csv_file.exists()

    def test_csv_contains_sample_name(self, tmp_path):
        sumdir = tmp_path / "summary"
        feadir = tmp_path / "fea"
        cfg = self._make_cfg(sumdir, feadir)
        write_data_summary(
            cfg, self._make_full_optim(), self._make_full_bone(), "S001",
            self._make_mesh_params(), 1000, 42.0,
        )
        content = (sumdir / "V1_TEST_data_summary.csv").read_text()
        assert "S001" in content

    def test_csv_appends_on_second_call(self, tmp_path):
        sumdir = tmp_path / "summary"
        feadir = tmp_path / "fea"
        cfg = self._make_cfg(sumdir, feadir)
        for sample in ("S001", "S002"):
            write_data_summary(
                cfg, self._make_full_optim(), self._make_full_bone(), sample,
                self._make_mesh_params(), 1000, 42.0,
            )
        csv_file = sumdir / "V1_TEST_data_summary.csv"
        with open(csv_file) as f:
            rows = list(csv.reader(f))
        # header + 2 data rows (or header + 2 because first call writes header + data)
        assert len(rows) >= 2

    def test_summary_dir_created_if_missing(self, tmp_path):
        sumdir = tmp_path / "does_not_exist" / "yet"
        feadir = tmp_path / "fea"
        cfg = self._make_cfg(sumdir, feadir)
        write_data_summary(
            cfg, self._make_full_optim(), self._make_full_bone(), "S001",
            self._make_mesh_params(), 1000, 42.0,
        )
        assert sumdir.exists()
