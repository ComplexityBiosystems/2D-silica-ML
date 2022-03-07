from typing import List, Tuple, Optional, TYPE_CHECKING
import numpy as np
import networkx as nx

if TYPE_CHECKING:
    from .base import SilicaGraph


def split_and_keep(s, sep):
    """Adapted from https://stackoverflow.com/a/34120478.
    """
    if not s:
        return ['']  # consistent with string.split()

    # Find replacement character that is not used in string
    # i.e. just use the highest available character plus one
    # Note: This fails if ord(max(s)) = 0x10FFFF (ValueError)
    p = chr(ord(max(s))+1)

    return s.replace(sep, p + sep).split(p)


def inflate_silica_graph(S: "SilicaGraph", thickness: float = 0.5):
    # top layer
    top = nx.Graph(S.copy())
    nx.relabel_nodes(
        top,
        mapping={
            i: f"{i}_top"
            for i in top.nodes
        },
        copy=False
    )

    # bottom layer
    bottom = nx.Graph(S.copy())
    nx.relabel_nodes(
        bottom,
        mapping={
            i: f"{i}_bottom"
            for i in bottom.nodes
        },
        copy=False
    )

    # move top and bottom nodes in Z direction
    for i in range(len(S)):
        top.nodes[f"{i}_top"]["position"] += (thickness / 2,)
        bottom.nodes[f"{i}_bottom"]["position"] += (-thickness / 2,)

    # combine
    G = nx.compose(top, bottom)

    # add middle oxygens below Si
    for i in range(len(S)):
        if S.nodes[i]["type"] == "Si":
            position = S.nodes[i]["position"] + (0,)
            G.add_node(f"{i}_middle", type="O", position=position)
            G.add_edge(f"{i}_top", f"{i}_middle")
            G.add_edge(f"{i}_middle", f"{i}_bottom")
    # add extra needed attributes
    G.size_x = S.size_x
    G.size_y = S.size_y
    G.thickness = thickness
    # relabel nodes sequentially from 0 to simplify lammps export
    nx.relabel_nodes(
        G,
        mapping={
            node: i
            for i, node in enumerate(G.nodes)
        },
        copy=False
    )
    return G


def average_pbc(points: List[Tuple[float, float]], size_x: float, size_y: float):
    """Compute the average of a set of points even if they go over PBC"""

    x0 = points[0]
    vectors_from_x0 = np.array([
        list(cyclic_vector(x0, v, size_x, size_y))
        for v in points[1:]
    ])
    av_x, av_y = np.array(x0) + np.sum(vectors_from_x0, axis=0) / len(points)
    average = (av_x % size_x, av_y % size_y)
    return average


def cyclic_vector(u, v, size_x, size_y):
    _, dx, dy = cyclic_distance(u, v, size_x, size_y)
    return (dx, dy)


def cyclic_distance(u, v, size_x, size_y):
    u_x, u_y = u
    v_x, v_y = v
    sx = np.sign(v_x - u_x)
    dx = np.abs(v_x - u_x)
    if dx > size_x / 2:
        sx *= -1
        dx = size_x - dx
    sy = np.sign(v_y - u_y)
    dy = np.abs(v_y - u_y)
    if dy > size_y / 2:
        sy *= -1
        dy = size_y - dy
    return np.sqrt(dx ** 2 + dy ** 2), sx*dx, sy*dy


def rotate_silicagraph_in_pbc(S, theta: Optional[float] = None):
    """Rotate a silica graph inside PBC"""

    positions = nx.get_node_attributes(S, "position")
    positions = np.array([list(pos) for pos in positions.values()])
    types = nx.get_node_attributes(S, "type")
    types = np.array([x for x in types.values()])

    new_positions = []
    new_types = []
    for horizontal in [-S.size_x, 0, S.size_x]:
        for vertical in [-S.size_y, 0, S.size_y]:
            for pos, type in zip(positions, types):
                new_pos = [pos[0] + horizontal, pos[1] + vertical]
                new_positions.append(new_pos)
                new_types.append(type)

    new_positions = np.array(new_positions)
    new_types = np.array(new_types)

    mean = np.mean(new_positions, axis=0)
    if theta is None:
        theta = np.random.uniform(0, 2*np.math.pi)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    rotated_positions = (new_positions - mean).dot(R) + mean

    size = 0.5 * (S.size_x + S.size_y)

    b1 = rotated_positions[:, 0] >= -size / 2
    b2 = rotated_positions[:, 0] < 1.5 * size
    b3 = rotated_positions[:, 1] >= -size / 2
    b4 = rotated_positions[:, 1] < 1.5 * size

    inside_box = np.all([b1, b2, b3, b4], axis=0)
    final_positions = rotated_positions[inside_box]
    final_types = new_types[inside_box]
    return final_positions, final_types
