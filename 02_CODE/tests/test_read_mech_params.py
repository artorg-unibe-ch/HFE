"""
Tests for hfe_utils/read_mech_params.py

Covers:
- remove_empty_entries_list
- __stiffness__
- __yield_point__
- datfilereader_6d (with temporary dat file)
- datfilereader_force (with temporary dat file)
- parse_and_calculate_stiffness_yield_force (integration, with temporary dat file)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import importlib
import textwrap
import tempfile
import os

import numpy as np
import pandas as pd
import pytest

import hfe_utils.read_mech_params as rmech

remove_empty_entries_list = rmech.remove_empty_entries_list
_stiffness = rmech.__dict__["__stiffness__"]
_yield_point = rmech.__dict__["__yield_point__"]
_calc_mech_props = rmech.__dict__["__calc_mech_props__"]


# ---------------------------------------------------------------------------
# remove_empty_entries_list
# ---------------------------------------------------------------------------


class TestRemoveEmptyEntriesList:
    def test_removes_empty_strings(self):
        data = ["", "", "1.0", "2.0", "3.0\n"]
        result = remove_empty_entries_list(data)
        assert "" not in result

    def test_strips_trailing_newline_from_last_element(self):
        data = ["", "1.0", "2.0", "3.0\n"]
        result = remove_empty_entries_list(data)
        assert result[-1] == "3.0"

    def test_returns_cleaned_list(self):
        data = ["", " ", "alpha", "beta", "gamma\n"]
        # only empty strings are removed, not whitespace
        result = remove_empty_entries_list(data)
        assert "" not in result
        assert result[-1] == "gamma"

    def test_single_element_list(self):
        data = ["value\n"]
        result = remove_empty_entries_list(data)
        assert result == ["value"]

    def test_no_empty_strings(self):
        data = ["1.0", "2.0", "3.0\n"]
        result = remove_empty_entries_list(data)
        assert result == ["1.0", "2.0", "3.0"]

    def test_multiple_empty_strings(self):
        data = ["", "", "", "a", "b\n"]
        result = remove_empty_entries_list(data)
        assert result == ["a", "b"]

    def test_typical_abaqus_split_line(self):
        """Simulate a line like '  1  1.0  2.0  3.0  4.0  5.0  6.0\n'.split(' ')"""
        line = "  1  1.0  2.0  3.0  4.0  5.0  6.0\n"
        parts = line.split(" ")
        result = remove_empty_entries_list(parts)
        assert "1.0" in result
        assert "6.0" in result
        # Last element should not have trailing newline
        assert not result[-1].endswith("\n")


# ---------------------------------------------------------------------------
# __stiffness__
# ---------------------------------------------------------------------------


class TestStiffness:
    def _make_df(self, rf3_vals, u3_vals):
        return pd.DataFrame({"RF3": rf3_vals, "U3": u3_vals})

    def test_basic_stiffness(self):
        df = self._make_df([100.0, 200.0, 300.0], [1.0, 2.0, 3.0])
        k, FZ, DZ = _stiffness(df)
        assert k == pytest.approx(100.0)

    def test_returns_arrays(self):
        df = self._make_df([50.0, 100.0], [0.5, 1.0])
        k, FZ, DZ = _stiffness(df)
        assert len(FZ) == 2
        assert len(DZ) == 2

    def test_zero_displacement_fallback(self):
        """When DZ[0] == 0, stiffness computed from slope between indices 1 and 2."""
        df = self._make_df([0.0, 50.0, 100.0], [0.0, 1.0, 2.0])
        k, FZ, DZ = _stiffness(df)
        # fallback: (100 - 50) / (2 - 1) = 50
        assert k == pytest.approx(50.0)

    def test_stiffness_with_large_forces(self):
        df = self._make_df([1000.0, 2000.0], [2.0, 4.0])
        k, FZ, DZ = _stiffness(df)
        assert k == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# __yield_point__
# ---------------------------------------------------------------------------


class TestYieldPoint:
    def test_basic_yield_found(self):
        """Yield point found when FZ drops below the 0.2%-offset line."""
        height = 10.0
        k = 100.0
        # Create a force-displacement curve where yield is clearly exceeded
        DZ = np.array([0.0, 0.1, 0.2, 0.3])
        FZ = np.array([0.0, 10.0, 20.0, 15.0])  # drops at index 3
        Fyield, disp_yield = _yield_point(height, k, FZ, DZ)
        assert Fyield > 0
        assert disp_yield > 0

    def test_no_yield_returns_large_value(self):
        """When no yield point found, returns 1e6 for both values."""
        height = 10.0
        k = 100.0
        # Monotonically increasing force — never crosses offset line
        DZ = np.array([0.0, 0.1, 0.2, 0.3])
        FZ = np.array([0.0, 50.0, 100.0, 150.0])
        Fyield, disp_yield = _yield_point(height, k, FZ, DZ)
        assert Fyield == pytest.approx(1e6)
        assert disp_yield == pytest.approx(1e6)

    def test_yield_point_values_are_finite(self):
        height = 30.6
        k = 500.0
        DZ = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
        FZ = np.array([0.0, 25.0, 50.0, 70.0, 60.0, 40.0])
        Fyield, disp_yield = _yield_point(height, k, FZ, DZ)
        assert np.isfinite(Fyield)
        assert np.isfinite(disp_yield)

    def test_yield_displacement_between_adjacent_points(self):
        """disp_yield should be between the two bounding DZ points where crossing occurs."""
        height = 10.0
        k = 100.0
        DZ = np.array([0.0, 0.1, 0.2, 0.3])
        FZ = np.array([0.0, 10.0, 20.0, 15.0])
        Fyield, disp_yield = _yield_point(height, k, FZ, DZ)
        # Crossing happens between DZ[2]=0.2 and DZ[3]=0.3
        assert 0.2 <= disp_yield <= 0.3


# ---------------------------------------------------------------------------
# datfilereader_6d — integration test with a temporary .dat file
# ---------------------------------------------------------------------------

def _build_dat_6d(u_data, rf_data, padding=("",) * 8):
    """
    Build lines for a minimal .dat file understood by datfilereader_6d.

    The trigger line (containing "U3") is at index 0.
    The U-values line is at index 3.
    The RF-values line is at index 12.
    """
    def fmt_data(vals):
        # Produce a line like "  1  1.0  2.0  3.0  4.0  5.0  6.0\n"
        parts = ["  1"] + [f"  {v}" for v in vals]
        return " ".join(parts) + "\n"

    lines = [
        " U1       U2       U3       UR1      UR2      UR3\n",  # 0 – trigger
        "\n",  # 1
        "\n",  # 2
        fmt_data(u_data),  # 3 – U values
        "\n",  # 4
        "\n",  # 5
        "\n",  # 6
        "\n",  # 7
        "\n",  # 8
        " RF1      RF2      RF3      RM1      RM2      RM3\n",  # 9
        "\n",  # 10
        "\n",  # 11
        fmt_data(rf_data),  # 12 – RF values
        "\n",  # 13
    ]
    return lines


class TestDatfilereader6d:
    def _write_temp_dat(self, lines):
        fd, path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        with open(path, "w") as f:
            f.writelines(lines)
        return path

    def test_returns_txt_filename(self):
        lines = _build_dat_6d(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        dat_path = self._write_temp_dat(lines)
        try:
            txt_path = rmech.datfilereader_6d(dat_path)
            assert txt_path.endswith(".txt")
            assert os.path.exists(txt_path)
        finally:
            for p in (dat_path, dat_path.replace(".dat", ".txt")):
                if os.path.exists(p):
                    os.remove(p)

    def test_output_file_contains_header(self):
        lines = _build_dat_6d(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        dat_path = self._write_temp_dat(lines)
        txt_path = dat_path.replace(".dat", ".txt")
        try:
            rmech.datfilereader_6d(dat_path)
            with open(txt_path) as f:
                content = f.read()
            assert "inc" in content
            assert "U1" in content
        finally:
            for p in (dat_path, txt_path):
                if os.path.exists(p):
                    os.remove(p)

    def test_empty_file_produces_empty_output(self):
        dat_path = self._write_temp_dat(["\n"])
        txt_path = dat_path.replace(".dat", ".txt")
        try:
            rmech.datfilereader_6d(dat_path)
            with open(txt_path) as f:
                lines_out = f.readlines()
            # Only header lines — no data rows
            data_lines = [
                l for l in lines_out if l.strip() and not l.startswith("*") and l.strip() != "0,"
            ]
            assert len(data_lines) == 0
        finally:
            for p in (dat_path, txt_path):
                if os.path.exists(p):
                    os.remove(p)


# ---------------------------------------------------------------------------
# datfilereader_force — integration test with a temporary .dat file
# ---------------------------------------------------------------------------

def _build_dat_force(u_data, rf_data, cf_data):
    """
    Build a minimal .dat file for datfilereader_force.

    Trigger at index 0, U at index 3, RF at index 12, CF at index 21.
    """
    def fmt_data(vals):
        parts = ["  1"] + [f"  {v}" for v in vals]
        return " ".join(parts) + "\n"

    lines = [
        " U1       U2       U3       UR1      UR2      UR3\n",  # 0 – trigger
        "\n",  # 1
        "\n",  # 2
        fmt_data(u_data),  # 3
        "\n",  # 4
        "\n",  # 5
        "\n",  # 6
        "\n",  # 7
        "\n",  # 8
        " RF1      RF2      RF3      RM1      RM2      RM3\n",  # 9
        "\n",  # 10
        "\n",  # 11
        fmt_data(rf_data),  # 12
        "\n",  # 13
        "\n",  # 14
        "\n",  # 15
        "\n",  # 16
        "\n",  # 17
        " CF1      CF2      CF3      CM1      CM2      CM3\n",  # 18
        "\n",  # 19
        "\n",  # 20
        fmt_data(cf_data),  # 21
        "\n",  # 22
    ]
    return lines


class TestDatfilereaderForce:
    def _write_temp_dat(self, lines):
        fd, path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        with open(path, "w") as f:
            f.writelines(lines)
        return Path(path)

    def test_returns_ref_nodedata_list(self):
        lines = _build_dat_force(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        )
        dat_path = self._write_temp_dat(lines)
        txt_path = dat_path.with_suffix(".txt")
        try:
            ref_nodedata = rmech.datfilereader_force(dat_path)
            assert isinstance(ref_nodedata, list)
            assert len(ref_nodedata) >= 1
        finally:
            for p in (dat_path, txt_path):
                if p.exists():
                    p.unlink()

    def test_output_data_row_contains_values(self):
        u = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        rf = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        cf = [13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
        lines = _build_dat_force(u, rf, cf)
        dat_path = self._write_temp_dat(lines)
        txt_path = dat_path.with_suffix(".txt")
        try:
            ref_nodedata = rmech.datfilereader_force(dat_path)
            # First row should contain our U3 value (3.0)
            first_row = ref_nodedata[0]
            assert "3.0" in first_row
        finally:
            for p in (dat_path, txt_path):
                if p.exists():
                    p.unlink()

    def test_output_txt_file_created(self):
        lines = _build_dat_force(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        dat_path = self._write_temp_dat(lines)
        txt_path = dat_path.with_suffix(".txt")
        try:
            rmech.datfilereader_force(dat_path)
            assert txt_path.exists()
        finally:
            for p in (dat_path, txt_path):
                if p.exists():
                    p.unlink()

    def test_empty_file_returns_empty_list(self):
        dat_path = self._write_temp_dat(["\n"])
        txt_path = dat_path.with_suffix(".txt")
        try:
            ref_nodedata = rmech.datfilereader_force(dat_path)
            assert ref_nodedata == []
        finally:
            for p in (dat_path, txt_path):
                if p.exists():
                    p.unlink()


# ---------------------------------------------------------------------------
# parse_and_calculate_stiffness_yield_force — integration
# ---------------------------------------------------------------------------


class TestParseAndCalculateStiffnessYieldForce:
    """Integration test: builds a multi-increment dat file and runs the full parser."""

    def _write_multi_increment_dat(self, increments):
        """
        Build a dat file with several increments.
        Each increment: U = [u*step, ...], RF = [f*step, ...], CF = [0, ...]
        """
        lines = []
        for u_vals, rf_vals, cf_vals in increments:
            block = _build_dat_force(u_vals, rf_vals, cf_vals)
            lines.extend(block)
            lines.append("\n")
        fd, path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        with open(path, "w") as f:
            f.writelines(lines)
        return Path(path)

    def test_stiffness_positive(self):
        # Create a linear force-displacement response plus a drop
        increments = [
            ([0.0, 0.0, s * 0.05, 0.0, 0.0, 0.0], [0.0, 0.0, s * 50.0, 0.0, 0.0, 0.0], [0.0] * 6)
            for s in range(1, 6)
        ]
        # Add a drop at the end
        increments.append(
            ([0.0, 0.0, 5 * 0.05, 0.0, 0.0, 0.0], [0.0, 0.0, 200.0, 0.0, 0.0, 0.0], [0.0] * 6)
        )
        dat_path = self._write_multi_increment_dat(increments)
        txt_path = dat_path.with_suffix(".txt")
        try:
            stiffness, yield_force, yield_disp, max_force, disp_at_max = (
                rmech.parse_and_calculate_stiffness_yield_force(dat_path, thickness=30.6)
            )
            assert stiffness > 0
            assert max_force > 0
        finally:
            for p in (dat_path, txt_path):
                if p.exists():
                    p.unlink()
