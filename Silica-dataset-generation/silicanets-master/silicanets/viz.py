from typing import Dict
import matplotlib.pyplot as plt
from .base import RingGraph, SilicaGraph
import numpy as np
import io


def show_silica(R: RingGraph, show_dual: bool = False, show_bbox: bool = True, ax: plt.Axes = None):
    ax_was_none = ax is None
    if ax_was_none:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    S = R.to_sio()
    ax.set_aspect(1)
    if show_dual:
        R.show(node_size=30, ax=ax)
    S.show(ax=ax)
    # bounding box
    if show_bbox:
        bbox = [
            [0, 0],
            [R.size_x, 0],
            [R.size_x, R.size_y],
            [0, R.size_y],
            [0, 0]
        ]
        ax.plot(*np.array(bbox).T, ls="dashed", c="0.5")
    # show fig if we didn't get axis
    if ax_was_none:
        fig.show()


def get_exp_heatmap_from_lammps_frame(
    frame: Dict,
    pixels=256,
    radius_Si=0.5,
    radius_O=0.3,
    photons_background=400000
):
    S = SilicaGraph.from_parsed_frame(frame)
    return get_square_exp_heatmap_from_silica(
        S=S,
        pixels=pixels,
        radius_Si=radius_Si,
        radius_O=radius_O,
        photons_background=photons_background
    )


def get_square_simple_heatmap_from_silica(
    S: SilicaGraph,
    pixels: int = 128,
    radius_Si: float = 50,
    radius_O: float = 0
):
    assert S.size_x >= S.size_y
    delta = (S.size_x - S.size_y) / 2

    # get node positions
    si_x, si_y = np.array([
        list(S.nodes[i]["position"])
        for i in S.nodes
        if S.nodes[i]["type"] == "Si"
    ]).T
    o_x, o_y = np.array([
        list(S.nodes[i]["position"])
        for i in S.nodes
        if S.nodes[i]["type"] == "O"
    ]).T

    # make a fig, remove axis etc
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_aspect(1)

    ax.scatter(si_x, si_y, s=radius_Si, c="0", marker="o")
    ax.scatter(o_x, o_y, s=radius_O, c="0", marker="o")

    ax.set_xlim(delta, S.size_x - delta)
    ax.set_ylim(0, S.size_y)
    ax.set_axis_off()
    # save figure into a bufferr
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', dpi=pixels / 10, )
    io_buf.seek(0)
    # get the pixels from the buffer
    t = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    t = np.reshape(t, newshape=(pixels, pixels, -1))
    t = np.max(255-t, axis=-1)
    tm = np.max(t)
    return t / tm


def get_square_exp_heatmap_from_silica(
    S: SilicaGraph,
    pixels=256,
    radius_Si=0.5,
    radius_O=0.3,
    photons_background=400000
):
    H = get_exp_heatmap_from_silica(
        S,
        pixel_size=S.size_y / pixels,
        radius_Si=radius_Si,
        radius_O=radius_O,
        photons_background=photons_background,
    )
    H_square = _make_square(H, pixels)
    return H_square


def _make_square(H, pixels):
    # make sure the image was stretched in x direction
    pixels_x, pixels_y = H.shape
    assert pixels_y == pixels, f"Expecting exactly {pixels} in y direction, found {pixels_y}"
    assert pixels_x >= pixels, f"Expecting at least {pixels} in x direction, found {pixels_x}"

    # make it square, centered in x axis
    _from = (pixels_x - pixels) // 2
    _to = _from + pixels
    H_square = H[_from:_to]

    assert H_square.shape == (pixels, pixels)
    return H_square


def get_exp_heatmap_from_silica(
    S: SilicaGraph,
    pixel_size: float = 0.01,
    radius_Si: float = 0.1,
    radius_O: float = 0.05,
    photons_background=20000
):
    bins_x = np.arange(0, S.size_x + pixel_size, pixel_size)
    bins_y = np.arange(0, S.size_y + pixel_size, pixel_size)
    radii_dict = {
        "Si": radius_Si,
        "O": radius_O
    }
    photons_per_atom_dict = {
        "Si": 800,
        "O": 100
    }
    radii_shift_dict = {
        "Si": 0,
        "O": 0.1
    }
    light = []
    # background
    light_bakcground = np.stack([
        np.random.uniform(0, S.size_x, size=photons_background),
        np.random.uniform(0, S.size_y, size=photons_background)
    ]).T
    light.append(light_bakcground)
    # atoms
    for i in range(len(S)):
        node = S.nodes[i]
        radius = radii_dict[node["type"]]
        photons_per_atom = photons_per_atom_dict[node["type"]]
        shift = radii_shift_dict[node["type"]]
        photons = light_profile(
            radius=radius, size=photons_per_atom, shift=shift)
        light.append(photons + np.array(node["position"]))

    light = np.concatenate(light)
    light_x, light_y = light.T
    H, _, _ = np.histogram2d(light_x, light_y, bins=(bins_x, bins_y))
    return H


def light_profile(radius: float, size: int, shift=0):
    x = np.random.normal(size=(size, 2), scale=radius)
    r = np.sqrt(np.sum(x**2, axis=1))
    return (x.T / r ** shift).T


def circle_uniform_profile(radius: float, size: int):
    x = np.random.uniform(-radius, radius, size=(2*size, 2))
    r = np.sqrt(np.sum(x**2, axis=1))
    x_inside = x[r <= radius]
    return x_inside[:size]
