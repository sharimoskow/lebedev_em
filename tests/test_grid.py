"""
test_grid.py — Tests for LebedevGrid3D and factory functions.
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lebedev_em.grid import (
    LebedevGrid3D,
    uniform_grid,
    symmetric_uniform_grid,
    optimal_geometric_1d,
    _E_CLUSTER_MAP,
    _H_CLUSTER_MAP,
    C000, C101, C110, C011,
)


class TestGridBasics:
    def test_even_grid_requirement(self):
        with pytest.raises(ValueError):
            LebedevGrid3D(np.linspace(0, 1, 4), np.linspace(0, 1, 5), np.linspace(0, 1, 5))

    def test_node_counts(self):
        grid = uniform_grid(4, 4, 4, 1.0, 1.0, 1.0)
        assert grid.Nx == 5
        assert grid.N_R + grid.N_P == grid.Nx * grid.Ny * grid.Nz

    def test_pr_partition(self):
        """Every node is in exactly one of P or R."""
        grid = uniform_grid(4, 6, 8, 1.0, 1.0, 1.0)
        for i in range(grid.Nx):
            for j in range(grid.Ny):
                for k in range(grid.Nz):
                    in_R = grid.R_idx[i, j, k] >= 0
                    in_P = grid.P_idx[i, j, k] >= 0
                    assert in_R != in_P, f"Node ({i},{j},{k}) fails P/R partition."

    def test_parity_rule(self):
        """R nodes must have odd (i+j+k), P nodes even."""
        grid = uniform_grid(6, 6, 6, 2.0, 2.0, 2.0)
        for i, j, k in grid.R_nodes:
            assert (i + j + k) % 2 == 1
        for i, j, k in grid.P_nodes:
            assert (i + j + k) % 2 == 0

    def test_sequential_index_roundtrip(self):
        """R_idx[R_nodes[seq]] == seq."""
        grid = uniform_grid(4, 4, 4, 1.0, 1.0, 1.0)
        for seq, (i, j, k) in enumerate(grid.R_nodes):
            assert grid.R_idx[i, j, k] == seq
        for seq, (i, j, k) in enumerate(grid.P_nodes):
            assert grid.P_idx[i, j, k] == seq


class TestClusterMaps:
    def test_e_cluster_map_coverage(self):
        """Every R-node type has exactly 3 distinct clusters assigned."""
        for node_type, clusters in _E_CLUSTER_MAP.items():
            assert len(set(clusters)) == 3, f"Node type {node_type}: {clusters}"

    def test_h_cluster_map_coverage(self):
        """Every P-node type has exactly 3 distinct clusters assigned."""
        for node_type, clusters in _H_CLUSTER_MAP.items():
            assert len(set(clusters)) == 3, f"Node type {node_type}: {clusters}"

    def test_e_cluster_array_shape(self):
        grid = uniform_grid(4, 4, 4, 1.0, 1.0, 1.0)
        assert grid.E_cluster.shape == (grid.N_R, 3)

    def test_each_cluster_appears_once_per_component_per_node_type(self):
        """For each component (Ex/Ey/Ez), each cluster appears at exactly 1 of the 4 node types."""
        for comp in range(3):
            clusters_seen = [_E_CLUSTER_MAP[t][comp] for t in _E_CLUSTER_MAP]
            # Should be a permutation of {0, 1, 2, 3}
            assert sorted(clusters_seen) == [0, 1, 2, 3], \
                f"Component {comp}: cluster assignments {clusters_seen}"

    def test_r_nodes_for_cluster_component(self):
        """The per-cluster node lists partition all R-nodes for each component."""
        grid = uniform_grid(6, 6, 6, 1.0, 1.0, 1.0)
        for comp in range(3):
            total = 0
            all_seqs = set()
            for c in (C000, C101, C110, C011):
                seqs = grid.r_nodes_for_cluster_component(c, comp)
                all_seqs.update(seqs)
                total += len(seqs)
            assert total == grid.N_R, f"Component {comp}: {total} ≠ {grid.N_R}"
            assert len(all_seqs) == grid.N_R


class TestBoundaryDetection:
    def test_boundary_nodes_exist(self):
        grid = uniform_grid(4, 4, 4, 1.0, 1.0, 1.0)
        bdy = grid.r_boundary_mask()
        assert bdy.sum() > 0

    def test_interior_exists(self):
        grid = uniform_grid(8, 8, 8, 1.0, 1.0, 1.0)
        bdy = grid.r_boundary_mask()
        assert (~bdy).sum() > 0


class TestOptimalGrid:
    def test_optimal_1d_monotone(self):
        xh = optimal_geometric_1d(k=5, h_min=0.01, L=1.0)
        assert np.all(np.diff(xh) > 0), "Optimal grid must be monotonically increasing."

    def test_optimal_1d_length(self):
        k = 6
        xh = optimal_geometric_1d(k=k, h_min=0.01, L=1.0)
        # optimal_geometric_1d returns the full interleaved primary+dual grid:
        # k+1 primary nodes interleaved with k dual midpoints → 2k+1 total.
        assert len(xh) == 2 * k + 1

    def test_symmetric_grid_even_Mx(self):
        from lebedev_em.grid import symmetric_optimal_grid
        z = np.linspace(-5, 5, 21)  # Mz=20, even ✓
        g = symmetric_optimal_grid(h_min=0.05, L=2.0, z=z, k=3)
        assert g.Mx % 2 == 0
        assert g.My % 2 == 0


class TestGridSummary:
    def test_summary_runs(self):
        grid = uniform_grid(4, 4, 4, 1.0, 1.0, 1.0)
        s = grid.summary()
        assert "LebedevGrid3D" in s
