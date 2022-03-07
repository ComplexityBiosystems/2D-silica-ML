"""Functions to analyze exeprimental data of 2D silica"""
import numpy as np
from scipy.spatial import Voronoi, Delaunay
from scipy.spatial import distance

from typing import List
from copy import copy

import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Polygon


class SilicaTessellation:
    def __init__(self, points: np.ndarray) -> None:
        self.points = points
        self._infer_bonds()
        self._find_regions()

    def show(self, ax=None, ring_colors_dict=None):
        # define some coloring if none given
        if ring_colors_dict is None:
            self._ring_colors_dict = defaultdict(
                lambda: "black",  # default color for all other cases
                {
                    4: "#c4622d",  # strong orange
                    5: "#dea971",  # pale orange
                    6: "0.8",      # neutral gray
                    7: "#73a4e6",  # pale blue
                    8: "#2f6bbd",  # mid blue
                    9: "#0a3866"   # strong blue
                }
            )
        # create axis if not give
        axwasnone = ax is None
        if axwasnone:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # scatter for original points
        ax.scatter(*self.points.T, s=10, c="0.2", zorder=10)
        for region in self.regions:
            xy = self.points[region + [region[0]]]
            ax.add_patch(
                Polygon(
                    xy=xy,
                    ec="0.2",
                    fc=self._ring_colors_dict[len(region)],
                    alpha=0.7)
            )

    def _find_regions(self):
        self.tri = Delaunay(self.points)
        self._left_triangles = list(copy(self.tri.simplices))
        self.regions = []
        while self._left_triangles:
            region = self._flood(
                init_triangle=self._left_triangles.pop()
            )
            self.regions.append(region)

        # remove duplicate regions (each region is found once for each triangle)
        unique_regions = [
            list(y)
            for y in set([tuple(x) for x in self.regions])
        ]
        self.regions = unique_regions

    def _flood(self, init_triangle) -> List[int]:
        # set of possible walls I still need to check
        to_visit = []
        # set of walls I have accepted
        walls = []
        # set of walls I have rejected
        no_walls = []
        # add three initial bonds to list of missing
        for bond in self._tri_to_bonds(init_triangle):
            to_visit.append(bond)

        # keep adding walls till no more to visit
        while to_visit:
            bond_to_check = to_visit.pop()
            if bond_to_check in self.bonds:
                walls.append(bond_to_check)
            else:
                no_walls.append(bond_to_check)
                # add neighbouring bonds if they have never been checked
                for new_bond in self._find_four_neighbouring_bonds(bond_to_check):
                    if new_bond not in walls and new_bond not in no_walls:
                        to_visit.append(new_bond)
        # order counter-clockwise
        region = list(set([x for y in walls for x in y]))
        center_x, center_y = self.points[region].mean(axis=0)
        reorder = np.argsort(np.apply_along_axis(lambda p: np.arctan2(
            p[1] - center_y, p[0] - center_x) + np.pi*2, axis=1, arr=self.points[region]))
        ordered_region = [region[x] for x in reorder]
        return ordered_region

    def _find_four_neighbouring_bonds(self, bond_to_check):
        two_triangles = []
        for triangle in self.tri.simplices:
            for bond in self._tri_to_bonds(triangle):
                if bond == bond_to_check:
                    two_triangles.append(triangle)
        four_bonds = []
        for triangle in two_triangles:
            for bond in self._tri_to_bonds(triangle):
                if bond != bond_to_check:
                    four_bonds.append(bond)
        return four_bonds

    def _tri_to_bonds(self, triangle: List[int]):
        return \
            set([triangle[0], triangle[1]]), \
            set([triangle[0], triangle[2]]), \
            set([triangle[1], triangle[2]]),

    def _infer_bonds(self):
        # creo oggetto self.voronoi
        self.voronoi = Voronoi(self.points)
        ridge_points = self.voronoi.ridge_points
        vertices = self.voronoi.vertices
        regions = self.voronoi.regions

        ridge_points_new = []

        # create new list of ridge points containing the pairs of points that share a bond, and its length
        for link in ridge_points:

            counter_0 = 0
            counter_1 = 0

            idx_region_0 = self.voronoi.point_region[link[0]]
            idx_region_1 = self.voronoi.point_region[link[1]]

            region_0 = regions[idx_region_0]
            region_1 = regions[idx_region_1]

            # check border stuff
            for i in region_0:
                if i == -1:
                    counter_0 = 1

            for i in region_1:
                if i == -1:
                    counter_1 = 1

            if counter_0 == 1 and counter_1 == 1:
                continue

            commonalities = list(set(region_0).intersection(region_1))

            vertice_0 = vertices[commonalities[0]]
            vertice_1 = vertices[commonalities[1]]

            ridge_lenght = distance.euclidean(vertice_0, vertice_1)

            ridge_points_new.append([link[0], link[1], ridge_lenght])

        self.bonds = []

        # per ogni punto vedo le coppie con dentro esso presenti in ridge_points_new e scelgo le 3 coppie con
        # ridge_lenght maggiore
        # funzione stupida per fare sort in base alla lunghezza del bonds
        def myFunc(e):
            return e[2]

        for i in range(len(self.points)):

            neighbour = list(filter(lambda x: i in x, ridge_points_new))

            neighbour.sort(key=myFunc, reverse=True)

            if len(neighbour) < 3:
                continue

            bonds_0 = [neighbour[0][0], neighbour[0][1]]
            bonds_1 = [neighbour[1][0], neighbour[1][1]]
            bonds_2 = [neighbour[2][0], neighbour[2][1]]

            self.bonds.append(set(bonds_0))
            self.bonds.append(set(bonds_1))
            self.bonds.append(set(bonds_2))
