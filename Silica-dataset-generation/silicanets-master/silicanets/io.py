from .utils import split_and_keep
import networkx as nx
from typing import TYPE_CHECKING, Union, Dict, List
from pathlib import Path
if TYPE_CHECKING:
    from .base import RingGraph
import pandas as pd
import numpy as np


def read_simulation_dump(dump_file: Union[str, Path]) -> List[Dict]:
    """Read the output of a multi-frame LAMMPS simulation."""
    dump_file = Path(dump_file)
    raw_text = dump_file.read_text()
    raw_frames = split_and_keep(raw_text, "ITEM: TIMESTEP")[1:]
    parsed_frames = []
    for frame in raw_frames:
        try:
            parsed_frames.append(parse_frame(frame))
        except:
            pass
    ratio_parsed_frames = len(parsed_frames) / len(raw_frames)
    if ratio_parsed_frames < 0.95:
        raise RuntimeError(
            f"Only parsed {ratio_parsed_frames*100}% of the frames.")
    return parsed_frames


def parse_frame(frame: str) -> Dict:
    """Parse a single frame from a LAMMPS dump file."""
    # split raw text into lines
    lines = [line.strip() for line in frame.split("\n")]
    # make sure we have all required headers
    headers_to_catch = [
        'ITEM: TIMESTEP',
        'ITEM: NUMBER OF ATOMS',
        'ITEM: BOX BOUNDS pp pp pp',
        'ITEM: ATOMS id type xu yu zu c_peratom c_stratom[1]'
    ]
    for header in headers_to_catch:
        assert header in lines, f"Missing header '{header}'"
    # get line numbers of headers
    header_positions = [
        i for i, line in enumerate(lines)
        if line in headers_to_catch
    ]
    # make sure headers are in corrrect position
    expected_header_positions = [0, 2, 4, 8]
    assert header_positions == expected_header_positions, f"Headers positions are {header_positions}, but expected {expected_header_positions}"

    # parse frame now that we know lines are ok
    parsed_frame = {}
    parsed_frame["timestep"] = int(lines[1])
    parsed_frame["number_of_atoms"] = int(lines[3])
    parsed_frame["box_bounds"] = [
        tuple(float(x) for x in line.split(" "))
        for line in lines[5:8]
    ]
    df = pd.DataFrame(
        [
            line.split()
            for line in lines[9:-1]
        ]
    )
    # set column names from header consistently
    df.columns = headers_to_catch[-1].split(" ")[2:]
    # add a column with timestep
    df["timestep"] = parsed_frame["timestep"]
    # convert all columns to floats or ints
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    parsed_frame["atoms"] = df
    return parsed_frame


def read_lammps_dump(dump_file: Union[str, Path]) -> dict:
    dump_file = Path(dump_file)
    lines = dump_file.read_text().split("\n")[9:-1]
    positions = {}
    for line in lines:
        i, x, y = line.strip().split()
        node_id = int(i) - 1
        positions[node_id] = (float(x), float(y))
    return positions


def get_lammps_string(
    R: "RingGraph",
    three_dim: bool = False
) -> str:
    # compose lammps string
    lammps_string = \
        write_header(R=R, three_dim=three_dim) + \
        write_atoms_section(R=R, three_dim=three_dim) + \
        write_masses_section(R=R, three_dim=three_dim)
    if not three_dim:
        lammps_string += \
            write_bonds_section(R=R, three_dim=three_dim) + \
            write_bond_coeffs_section(R=R, three_dim=three_dim)
    return lammps_string


def write_atoms_section(R: "RingGraph", three_dim: bool = False) -> str:
    """Create string representation of nodes of a lattice, LAMMPS syntax."""
    header = """
Atoms

"""
    # all nodes are of type 1
    positions_dict = nx.get_node_attributes(R, "position")
    types_dict = nx.get_node_attributes(R, "type")
    conversion = {"Si": 1, "O": 2}

    if three_dim:
        lines = [
            f"{node + 1} 1 {conversion[types_dict[node]]} 0 {x} {y} {z}"
            for node, (x, y, z) in positions_dict.items()
        ]
    else:
        lines = [
            f"{node + 1} 1 1 {x} {y} 0"
            for node, (x, y) in positions_dict.items()
        ]

    section = header + "\n".join(lines) + "\n"
    return section


def write_masses_section(R: "RingGraph", three_dim: bool = False) -> str:
    """create a string represetnation of the masses info in LAMMPS syntax"""
    header = """
Masses

"""
    if three_dim:
        num_atom_types = 2
    else:
        num_atom_types = 1

    lines = [
        f"{atom_type} 1.0"
        for atom_type in range(1, num_atom_types + 1)
    ]
    section = header + "\n".join(lines) + "\n"
    return section


def write_bonds_section(R: "RingGraph", three_dim: bool = False) -> str:
    """Create string representation of bonds of a lattice using LAMMPS syntax."""
    header = """
Bonds

"""
    lines = [
        f"{idx + 1} {idx + 1} {i + 1} {j + 1}"
        for idx, (i, j) in enumerate(R.edges)
    ]
    section = header + "\n".join(lines) + "\n"
    return section


def write_bond_coeffs_section(R: "RingGraph", three_dim: bool = False) -> str:
    """Create string representation of bonds resting legnths of a lattice using LAMMPS syntax."""
    header = """
Bond Coeffs

"""
    lines = [
        # according to main paper
        # f"{idx + 1} {2 / R[i][j]['resting_length'] ** 2} {R[i][j]['resting_length']}"
        # according to original paper
        f"{idx + 1} 1 {R[i][j]['resting_length']}"
        for idx, (i, j) in enumerate(R.edges)
    ]
    section = header + "\n".join(lines) + "\n"
    return section


def write_header(R: Union[nx.Graph, "RingGraph"], three_dim: bool = False) -> str:
    """create string for the header in lammps format"""
    num_atoms = R.number_of_nodes()
    num_bonds = R.number_of_edges()
    num_atom_types = 1

    xlo, xhi = (0, R.size_x)
    ylo, yhi = (0, R.size_y)
    zlo, zhi = (-100, 100)

    if three_dim:
        # case Si-O
        num_atom_types = 2
        num_bonds = 0

    header = f"""
{num_atoms} atoms
{num_bonds} bonds
0 angles
0 dihedrals
0 impropers

{num_atom_types} atom types
{num_bonds} bond types
0 angle types
0 dihedral types
0 improper types

{xlo} {xhi} xlo xhi
{ylo} {yhi} ylo yhi
{zlo} {zhi} zlo zhi
"""
    return header
