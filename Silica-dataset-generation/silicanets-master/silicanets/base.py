from . import __LAMMPS_MINIMIZE_FIRE__
from .io import get_lammps_string
from .io import read_lammps_dump
from .utils import average_pbc, cyclic_distance

from typing import Tuple
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import gamma
from collections import defaultdict, Counter
import numpy as np

from pathlib import Path
from typing import Union, Dict
import subprocess
from temppathlib import TmpDirIfNecessary
from itertools import combinations

import pandas as pd
from scipy.stats import linregress
import copy


class RingGraph(nx.Graph):
    """A graph representing a dual 2D silica configuration."""

    def __init__(self, n_rows: int, n_cols: int, radius: float) -> None:
        """Create a ring graph with initial config hexagoanl lattice and PBC."""
        super().__init__()
        assert n_rows % 2 == 0, "Must have even number of rows for simpler boundary conditions"
        col_distance = radius
        row_distance = np.sqrt(3/4) * radius
        self.size_x = col_distance * n_cols
        self.size_y = row_distance * n_rows
        self.radius = radius
        # add points forming a hexagoanl lattice
        points = []
        y_start = row_distance / 4
        for row_idx, y in enumerate(np.arange(y_start, self.size_y, row_distance)):
            x_start = col_distance / 4 + (row_idx % 2) * col_distance / 2
            for x in np.arange(x_start, self.size_x, col_distance):
                points.append([x, y])
        # add edges when distance is equal to radius
        edges = []
        for i, u in enumerate(points):
            for j, v in enumerate(points):
                if i > j:
                    distance, _, _ = self._cyclic_distance(
                        u=u, v=v)
                    if np.abs(distance - radius) / radius < 1e-3:
                        edges.append([i, j])

        # create the graph. nodes represent rings
        for i, (x, y) in enumerate(points):
            self.add_node(i, position=(x, y))
        for i, j in edges:
            self.add_edge(i, j)

        # verify all degrees are 6 initially
        assert set(k for n, k in self.degree) == {6}

        # compute resting lengths, store as edge properties
        # notice this cannot be done before having added all edges
        # because degrees matter
        self._update_resting_lengths()

    def to_lammps(self, path_to_output: Union[str, Path], force: bool = False):
        if isinstance(path_to_output, str):
            path_to_output = Path(path_to_output)
        if path_to_output.exists() and not force:
            raise RuntimeError(
                "You are trying to overwrite an existing file. Make sure and then use --force option")
        lammps_str = get_lammps_string(R=self)
        path_to_output.write_text(lammps_str)

    def to_sio(self) -> "SilicaGraph":
        """Get the bipartite Si-O graph that this ring graph is representing."""
        triangles = [
            tuple(sorted(clique))
            for clique in nx.enumerate_all_cliques(self)
            if len(clique) == 3
        ]
        num_nodes_si = len(triangles)
        positions_si = {
            i: average_pbc(
                [
                    list(self.nodes[p]["position"])
                    for p in list(triangle)
                ],
                size_x=self.size_x,
                size_y=self.size_y
            )
            for i, triangle in enumerate(triangles)
        }
        # create graph, add Sis at centers of triangles
        S = SilicaGraph(size_x=self.size_x,
                        size_y=self.size_y)
        S.add_nodes_from(range(num_nodes_si), type="Si")
        nx.set_node_attributes(S, positions_si, "position")
        # add Os at middle of
        next_o_index = num_nodes_si
        for i, j in combinations(range(num_nodes_si), r=2):
            a, b, c = triangles[i]
            u, v, w = triangles[j]
            # case triangles share an edge
            if len({a, b, c, u, v, w}) == 4:
                o_index = next_o_index
                # position of the node
                u = S.nodes[i]["position"]
                v = S.nodes[j]["position"]
                S.add_node(
                    o_index,
                    type="O",
                    position=average_pbc([u, v], self.size_x, self.size_y)
                )
                # link the oxygen to the two silica
                S.add_edge(i, o_index)
                S.add_edge(j, o_index)
                # move the counter
                next_o_index += 1
        return S

    def set_target(self, variance: float, alpha=float):
        assert variance > 0, "Variance of target ring distribution must be positive"
        # set parameters for a gamma with mean 6 and variance variance
        mu = 6
        self.target_variance = variance
        self.target_ring_distribution = gamma(
            scale=variance / mu, a=mu ** 2 / variance).pdf
        # set alpha
        self.target_alpha = alpha
        # set ring range
        # consider only ring sizes that have expected rings > 0
        self.ring_range = [
            k
            for k in range(4, 20)
            if self.target_ring_distribution(k) * len(self) > 1
        ]
        self.target_ring_normalization = np.sum(
            self.target_ring_distribution(k) for k in self.ring_range
        )

    def mcmc(self, temperature: float, max_it: int, relax_every: int = 1) -> None:
        for it in range(max_it):
            # print current state of affairs
            current_obj = self._compute_objective_function()
            print(f"it: {it}, obj: {current_obj}")
            # make the switch
            edge = self._get_random_edge()
            new_edge = self.swap_edge(edge)
            new_obj = self._compute_objective_function()
            # compute prob of having done it
            delta_obj = new_obj - current_obj
            prob_switch = min(1, np.exp(-delta_obj / temperature))
            # undo if necessary
            if np.random.uniform() > prob_switch:
                _ = self.swap_edge(new_edge)
            # if swap was not reverted, relax once in a while
            # too often -> simu slow
            # too seldom -> relax ends up failing (but why?? it should not!)
            # of course its easier to relax at each step, we have a better i.c.
            elif it % relax_every == 0:
                self.relax()

    def swap_edge(self, edge: Tuple[int, int], method="degree") -> Tuple[int, int]:
        # find neighbours
        u, v = edge
        common_neighbors = list(nx.common_neighbors(self, u, v))
        # case switch is ok
        if self._verify_switch(edge, common_neighbors, method="degree"):
            self.remove_edge(u, v)
            self.add_edge(*common_neighbors)
            self._update_resting_length(*common_neighbors)
            return common_neighbors
        # case border
        else:
            return edge

    def relax(self, method="lammps") -> None:
        """Relax the dual lattice using harmonic ring-ring interactions."""
        # get position of nodes
        if method == "lammps":
            nodes_positions = self._relax_lammps()
        elif method == "python":
            nodes_positions == self._relax_python()
        else:
            raise RuntimeError("relaxation engine not recognized")

        # update position of nodes
        nx.set_node_attributes(self, nodes_positions, "position")

    def show(self, ax=None, node_color=None, **kwargs):
        ax_was_none = (ax is None)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # compute list of edges that do not go over PBC
        edgelist = []
        for (u, v) in self.edges:
            u_x, u_y = nx.get_node_attributes(self, "position")[u]
            v_x, v_y = nx.get_node_attributes(self, "position")[v]
            draw_distance = np.sqrt((v_x - u_x) ** 2 + (v_y - u_y) ** 2)
            real_distance, _, _ = self._cyclic_distance((u_x, u_y), (v_x, v_y))
            if np.isclose(real_distance, draw_distance):
                edgelist.append((u, v))

        # by default color by degree
        if node_color is None:
            node_color = [self.degree[i] for i in range(len(self))]
        nx.draw_networkx_nodes(
            G=self,
            pos=nx.get_node_attributes(self, "position"),
            ax=ax,
            node_color=node_color,
            **kwargs
        )
        nx.draw_networkx_edges(
            G=self,
            pos=nx.get_node_attributes(self, "position"),
            edgelist=edgelist,
            ax=ax,
        )
        if ax_was_none:
            ax.set_aspect(1)
            fig.show()

    def rescale(self, factor: float = 1) -> "RingGraph":
        """
        Rescale the current graph by a scale factor.

        Use this method to get the correct lengthg units.
        Notice that relax works well with radius 1.

        Returns
        -------
        RingGraph
            Copy of self with lengths rescaled by factor.
        """
        RR = copy.deepcopy(self)
        # rescale internal variables
        RR.size_x = self.size_x * factor
        RR.size_y = self.size_y * factor
        RR.radius = self.radius * factor
        # rescale node positions
        for i in RR.nodes:
            x, y = RR.nodes[i]["position"]
            RR.nodes[i]["position"] = (x * factor, y * factor)
        # rescale resting lengths
        for (u, v) in RR.edges:
            RR.edges[(u, v)]["resting_length"] *= factor
        return RR

    # PRIVATE METHODS
    def _cyclic_distance(self, u, v):
        return cyclic_distance(u=u, v=v, size_x=self.size_x, size_y=self.size_y)

    def _relax_lammps(self):
        # create tmp dir
        with TmpDirIfNecessary(path=None) as lammps_relax_dir:
            # with TmpDirIfNecessary("/tmp/lammps_tmp") as lammps_relax_dir:
            # do something with the operation directory
            lammps_relax_path = lammps_relax_dir.path
            self.to_lammps(lammps_relax_path / "test.lammps", force=True)
            # execute script
            process = subprocess.run(
                args=[
                    f"cd {lammps_relax_path} ;"
                    f"cp {__LAMMPS_MINIMIZE_FIRE__} . ;"
                    f"lmp_serial -in {Path(__LAMMPS_MINIMIZE_FIRE__).name};"
                ],
                shell=True,
                capture_output=True,
                executable="/bin/bash",
            )
            # find latest dump
            node_positions = read_lammps_dump(
                lammps_relax_path / "particles.dat")
            return node_positions

    def _relax_python(self, use_jacobian=False, **kwargs) -> dict:
        # mimize harmonic ring-ring potential
        fun, dfun = self._energy_function_factory()

        # initial condition and bounds
        x0 = []
        for i in range(len(self)):
            x0.append(nx.get_node_attributes(self, "position")[i][0])
        for i in range(len(self)):
            x0.append(nx.get_node_attributes(self, "position")[i][1])

        # use the jacobian or not
        if use_jacobian:
            jac = dfun
        else:
            jac = "2-point"

        # minimization
        self.result = minimize(fun=fun, x0=x0, jac=jac, **kwargs)

        # copy minimization result back into node positions
        xs: np.ndarray = self.result.x[: len(self)]
        ys: np.ndarray = self.result.x[len(self):]
        nodes_positions = {
            i: (xs[i], ys[i])
            for i in range(len(self))
        }
        return nodes_positions

    def _update_resting_lengths(self):
        for (i, j) in self.edges:
            self._update_resting_length(i, j)

    def _update_resting_length(self, i, j):
        invtan_i = 1 / np.tan(np.math.pi / self.degree(i))
        invtan_j = 1 / np.tan(np.math.pi / self.degree(j))
        resting_length = (self.radius / 2) * \
            (invtan_i + invtan_j) / 1.7320508075688776
        # adding edges that already exist just modifies them
        self.add_edge(i, j, resting_length=resting_length)

    def _verify_switch(self, edge, common_neighbors, method="two_neighbours"):
        if method == "two_neighbours":
            return self._verify_switch_two_neighbours(common_neighbors)
        elif method == "degree":
            return self._verify_swithch_degree(edge, common_neighbors)

    def _verify_switch_two_neighbours(self, common_neighbors):
        return len(common_neighbors) == 2

    def _verify_swithch_degree(self, edge, common_neighbors):
        if len(common_neighbors) != 2:
            return False
        u, v = edge
        nu, nv = common_neighbors
        # old nodes loss one neighbour
        k_u = self.degree(u) - 1
        k_v = self.degree(v) - 1
        # new nodes gain one neighbour
        k_nu = self.degree(nu) + 1
        k_nv = self.degree(nv) + 1
        new_degrees = np.array([k_u, k_v, k_nu, k_nv])
        assert len({u, v, nu, nv}) == 4
        # new config cannot leave anyone outside this limits
        if min(new_degrees) < 4 or max(new_degrees) > 10:
            return False
        else:
            return True

    def _get_ring_distribution(self):
        """Compute the distribuiton of degrees."""
        _, degrees = zip(*list(self.degree))
        ps = {a: b / len(degrees) for a, b in Counter(degrees).items()}
        # make defaultdict to add zeros as missing keys
        ps_dd = defaultdict(int, ps)
        return ps_dd

    def _get_alpha(self):
        # get degrees and average nn degrees
        df = pd.DataFrame(nx.average_degree_connectivity(
            self).items(), columns=["degree", "av_nn_deg"])
        # isolte (1-alpha in AW equation)
        # so that left = (1-alpha) * right
        mu = df.degree.var()
        right = 6*(df.degree - 6)
        left = df.av_nn_deg * df.degree - 36 - mu
        # do the fit
        alpha = 1 - linregress(right, left).slope
        return alpha

    def _compute_objective_function(self, k_scaling: float = 10):

        # alpha piece
        real_alpha = self._get_alpha()
        obj_alpha = k_scaling * np.abs(real_alpha - self.target_alpha)

        # ring distribution piece
        # divide by normalization constant
        # because ring sizes that bring less than one ring on average
        # are ignored
        real_ps = self._get_ring_distribution()
        target_ps = {
            n: self.target_ring_distribution(
                n) / self.target_ring_normalization
            for n in self.ring_range
        }
        obj_ring_distro = np.sum([
            np.abs(real_ps[n] - target_ps[n]) / target_ps[n]
            for n in self.ring_range
        ])
        # variance piece
        real_var = self._get_degrees_array().var()
        obj_variance = np.abs(
            real_var - self.target_variance) / self.target_variance

        # full objective function
        objective = obj_alpha + obj_variance + obj_ring_distro
        return objective

    def _get_degrees_array(self):
        _, degrees = zip(*list(self.degree))
        return np.array(degrees)

    def _get_random_edge(self):
        return list(self.edges)[np.random.randint(len(self.edges))]

    def _stats(self):
        real_alpha = self._get_alpha()
        print(f"alpha (target): {real_alpha} ({self.target_alpha})")

        real_var = self._get_degrees_array().var()
        print(f"deg_var (target): {real_var} ({self.target_variance})")

        # ring distribution piece
        real_ps = self._get_ring_distribution()
        target_ps = {
            n: self.target_ring_distribution(
                n) / self.target_ring_normalization
            for n in self.ring_range
        }
        print(f"n deg_prob_n (target)")
        for n in self.ring_range:
            print(f"{n} {real_ps[n]} {target_ps[n]}")


class SilicaGraph(nx.Graph):
    """A bipartite graph representing a Si-O configuration in 2D."""

    def __init__(self, size_x: float, size_y: float):
        super().__init__()
        self.size_x = size_x
        self.size_y = size_y

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def from_parsed_frame(parsed_frame: Dict, translation: Tuple[float, float] = (0, 0)) -> "SilicaGraph":
        # get top layer only
        df = parsed_frame["atoms"].copy()
        # if not x, y, z columns, maybe xu, yu, zu
        if "x" not in df.columns:
            assert "xu" in df.columns
            assert "yu" in df.columns
            assert "zu" in df.columns
            new_cols = [
                x.replace("xu", "x").replace("yu", "y").replace("zu", "z")
                for x in df.columns
            ]
            df.columns = new_cols

        sdf = df.loc[df.z > 1]
        # make sure we are close to 5/12 = 41.66% atoms in top layer
        frac_atoms_in_top_layer = len(sdf) / len(df)
        assert np.abs(frac_atoms_in_top_layer - 5 /
                      12) < 0.03, f"Only {len(sdf)} out of {len(df)} atoms in the top layer."

        # determine boundary conditions
        # since SilicaGraph works with (0, size_x), (0, size_y)
        # box, we must shift positions
        xmin, xmax = parsed_frame["box_bounds"][0]
        ymin, ymax = parsed_frame["box_bounds"][1]
        size_x = xmax - xmin
        size_y = ymax - ymin
        # shift positions accordingly
        df.x -= xmin
        df.y -= ymin
        # now happily apply the translation
        df.x = df.x.apply(lambda t: (t + translation[0]) % size_x)
        df.y = df.y.apply(lambda t: (t + translation[1]) % size_y)

        # translate type to strings
        df["type_str"] = df.type.apply(lambda x: {1: "Si", 2: "O"}[x])
        # create graph
        S = SilicaGraph(size_x=size_x, size_y=size_y)
        for i, (_, row) in enumerate(df.loc[df.z > 1].iterrows()):
            S.add_node(
                i,
                type=row["type_str"],
                position=(row["x"], row["y"])
            )
        return S

    def show(self, ax=None, node_color=None, **kwargs):
        ax_was_none = (ax is None)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # compute list of edges that do not go over PBC
        edgelist = []
        for (u, v) in self.edges:
            u_x, u_y = nx.get_node_attributes(self, "position")[u]
            v_x, v_y = nx.get_node_attributes(self, "position")[v]
            draw_distance = np.sqrt((v_x - u_x) ** 2 + (v_y - u_y) ** 2)
            real_distance, _, _ = self._cyclic_distance((u_x, u_y), (v_x, v_y))
            if np.isclose(real_distance, draw_distance):
                edgelist.append((u, v))

        # by default color by type
        color_code = {
            "Si": "#205ad6",
            "O": "#e61c1c"
        }
        size_code = {
            "Si": 120,
            "O": 50
        }

        node_color = [
            color_code[self.nodes[i]["type"]]
            for i in range(self.number_of_nodes())
        ]
        node_size = [
            size_code[self.nodes[i]["type"]]
            for i in range(self.number_of_nodes())
        ]

        nx.draw_networkx_nodes(
            G=self,
            pos=nx.get_node_attributes(self, "position"),
            ax=ax,
            node_color=node_color,
            node_size=node_size,
            **kwargs
        )
        nx.draw_networkx_edges(
            G=self,
            pos=nx.get_node_attributes(self, "position"),
            edgelist=edgelist,
            edge_color="0.5",
            width=5,
            ax=ax,
        )
        if ax_was_none:
            ax.set_aspect(1)
            fig.show()

        # PRIVATE METHODS
    def _cyclic_distance(self, u, v):
        return cyclic_distance(u=u, v=v, size_x=self.size_x, size_y=self.size_y)
