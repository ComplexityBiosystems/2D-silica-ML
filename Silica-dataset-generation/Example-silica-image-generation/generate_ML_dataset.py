import fire
from pathlib import Path
from typing import Union, Tuple
import numpy as np
from imageio import imsave

from silicanets.io import read_simulation_dump
from silicanets.viz import (
    get_square_exp_heatmap_from_silica,
    get_square_simple_heatmap_from_silica,
)
from silicanets.base import SilicaGraph
from uuid import uuid4

import pandas as pd


def parse_bondbreak_file(bondbreak_file: Path) -> Tuple[float, float, float]:
    """
    Read and parse the bondbreak.dat file.

    [from RG email 27Dec 2020]
    Note that the file bondbreak.dat in each folder contains a first line with
    just the rupture strain, and a second line with the information about the
    oxygen atom involved in the breaking: its ID, energy, the ID of the two
    neighboring Si atoms, and their distance (the largest distance is the broken
    bond).


    Parameters
    ----------
    bondbreak_file : Path
        Path to the bondbreak file.

    Returns
    -------
    strain, id_o, id_si: Tuple[float, float, float]
        Strain and atom ids of the bond.
    """
    # read two lines
    strain_line, location_line = bondbreak_file.read_text().strip().split("\n")
    # first line is the rupture strain
    strain = float(strain_line)
    # second line has ids of two possible bonds
    _O, _E, _S1, _B1, _S2, _B2 = location_line.split(" , ")
    assert _O.find("O: ") == 0, f"Format specifier for {_O} is not 'O: '"
    assert _E.find("E= ") == 0, f"Format specifier for {_E} is not 'E= '"
    assert _S1.find("Si1: ") == 0, f"Format specifier for {_S1} is not 'Si1: '"
    assert _B1.find("BL1= ") == 0, f"Format specifier for {_B1} is not 'BL1= '"
    assert _S2.find("Si2: ") == 0, f"Format specifier for {_S2} is not 'Si2: '"
    assert _B2.find(
        "BL2 = ") == 0, f"Format specifier for {_B2} is not 'BL2 = '"
    id_o = int(_O.replace("O: ", ""))
    bl1 = float(_B1.replace("BL1= ", ""))
    bl2 = float(_B2.replace("BL2 = ", ""))
    # need to decide which bond to keep depending on bondlengths
    if bl1 > bl2:
        id_si = int(_S1.replace("Si1: ", ""))
    else:
        id_si = int(_S2.replace("Si2: ", ""))

    return strain, id_o, id_si


def check_run(path_to_run: Path):
    """Make sure the run has the correct files."""
    # make sure all needed files exist
    needed_files = [
        "sio2_1_initial.dump",
        "sio2_2_aqs.dump",
        "sio2_3_bondbreak_1.dump",
        "sio2_4_fractured.dump",
        "bondbreak.dat",
    ]
    for filename in needed_files:
        file = path_to_run / filename
        if not file.exists():
            return False
        if not file.stat().st_size > 0:
            return False

    # make sure we do not have any forbidden file
    forbiden_files = [
        "sio2_2_aqs.is_broken"
    ]
    for filename in forbiden_files:
        file = path_to_run / filename
        if file.exists():
            return False

    # make sure we have one and only one in this set
    just_one_of = [
        "sio2_3_bondbreak_1.is_broken",
        "sio2_3_bondbreak_2.is_broken",
        "sio2_3_bondbreak_3.is_broken",
    ]
    how_many = np.sum([
        (path_to_run / filename).exists()
        for filename in just_one_of
    ])
    if how_many != 1:
        return False

    return True


def parse_run(
    path_to_run: Union[str, Path],
    path_to_images_dir: Union[str, Path],
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    translation: Union[Tuple[float, float], str] = (0, 0),
    when: str = "initial",
    mode: str = "blurry",
    pixels: int = 256,
) -> None:
    # treat as paths
    path_to_run = Path(path_to_run)
    path_to_images_dir = Path(path_to_images_dir)

    # validate params
    valid_whens = ["initial"]
    #valid_whens = ["initial", "aqs", "bondbreak", "fractured"]

    valid_modes = ["simple", "blurry"]
    assert mode in valid_modes, f"Parameter 'mode' must be one of {valid_modes}"

    # make sure output dir exists
    assert path_to_images_dir.exists(), f"Output dir for images does not exist"

    # make sure image to be created does not exist
    run_name = path_to_run.name
    image_name = str(uuid4()) + ".png"
    path_to_image = path_to_images_dir / image_name
    assert (
        not path_to_image.exists()
    ), f"Image to be created {path_to_image} already exists! refusing to overwrite"

    # determine which dump we are using
    assert when in valid_whens, f"Parameter 'when' must be one of {valid_whens}"
    dump_file = {
        "initial": path_to_run / "sio2_1_initial.dump",
        "aqs": path_to_run / "sio2_2_aqs.dump",
        "bondbreak": path_to_run / "sio2_3_bondbreak.dump",
        "fractured": path_to_run / "sio2_4_fractured.dump",
    }[when]

    # get targets
    bondbreak_file = path_to_run / "bondbreak.dat"
    strain, id_o, id_si = parse_bondbreak_file(bondbreak_file)

    # read frames from dump
    frames = read_simulation_dump(dump_file)
    assert len(frames) == 1
    frame = frames[0]

    # make sure box is positioned at 0,0
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = frame["box_bounds"]
    assert xlo == 0, f"I was expecting bbox_x to start at 0"
    assert ylo == 0, f"I was expecting bbox_y to start at 0"

    # find position of broken bond
    df = frame["atoms"]
    x1 = df.loc[df.id == id_o, "xu"].values[0]
    y1 = df.loc[df.id == id_o, "yu"].values[0]
    x2 = df.loc[df.id == id_si, "xu"].values[0]
    y2 = df.loc[df.id == id_si, "yu"].values[0]

    # take care of pbc when averaging
    dx = x2 - x1
    dy = y2 - y1
    if dx > 0.5 * xhi:
        dx = xhi - dx
    if dy > 0.5 * yhi:
        dy = yhi - dy
    x_angstrom = x1 + 0.5 * dx
    y_angstrom = y1 + 0.5 * dy

    # verify or set translation
    if isinstance(translation, tuple):
        assert xlo <= translation[0] <= xhi
        assert ylo <= translation[1] <= yhi
    elif translation == "random":
        translation = (np.random.uniform(xlo, xhi),
                       np.random.uniform(ylo, yhi))
    else:
        raise RuntimeError(f"Unrecognized value in 'translation arg'")

    # save original break location
    original_x_angstrom = np.copy(x_angstrom)
    original_y_angstrom = np.copy(y_angstrom)

    # shift target by translation
    x_angstrom += translation[0]
    y_angstrom += translation[1]

    # wrap target
    x_angstrom = xlo + x_angstrom % (xhi - xlo)
    y_angstrom = ylo + y_angstrom % (yhi - ylo)

    # make sure target is now inside box, properly
    assert xlo <= x_angstrom < xhi, f"Target is outside the bbox, check wrapping"
    assert ylo <= y_angstrom < yhi, f"Target is outside the bbox, check wrapping"

    # load silicagraph
    S = SilicaGraph.from_parsed_frame(frame, translation=translation)

    if mode == "blurry":
        heatmap = get_square_exp_heatmap_from_silica(S=S, pixels=pixels)
        # rescale into correct integer range
        # rotate and flip to match standard image storing convention
        im = np.flipud((heatmap / 100 * 255).astype(np.uint8).T)
    elif mode == "simple":
        heatmap = get_square_simple_heatmap_from_silica(S=S, pixels=pixels)
        im = np.flipud((heatmap * 255).astype(np.uint8).T)
    else:
        raise ValueError(f"Unknown mode")

    # finally apply flips to image and to target
    if vertical_flip:
        im = np.flipud(im)
        y_angstrom = yhi - y_angstrom
    if horizontal_flip:
        im = np.fliplr(im)
        x_angstrom = xhi - x_angstrom

    # save image
    imsave(path_to_image, im=im)

    # write target to stdout
    metadata_dict = {
        "image": path_to_image,
        "run": run_name,
        "box_size_x": xhi,
        "box_size_y": yhi,
        "lammps_file": dump_file,
        "horizontal_flip": horizontal_flip,
        "vertical_flip": vertical_flip,
        "break_strain": strain,
        "absolute_translation_x": translation[0],
        "absolute_translation_y": translation[1],
        "absolute_break_location_x": x_angstrom,
        "absolute_break_location_y": y_angstrom,
        "absolute_original_break_location_x": original_x_angstrom,
        "absolute_original_break_location_y": original_y_angstrom,
        "relative_translation_x": translation[0] / xhi,
        "relative_translation_y": translation[1] / yhi,
        "relative_break_location_x": x_angstrom / xhi,
        "relative_break_location_y": y_angstrom / yhi,
        "relative_original_break_location_x": original_x_angstrom / xhi,
        "relative_original_break_location_y": original_y_angstrom / yhi
    }
    return metadata_dict


def parse_dataset(
    glob_to_runs: str,
    path_to_images_dir: Union[str, Path],
    output_metadata: str,
    num_translations: int = 4,
    when: str = "initial",
    mode: str = "blurry",
    pixels: int = 256,
) -> None:
    metadata_listdict = []
    for path_to_run in Path(".").glob(glob_to_runs):
        if check_run(path_to_run=path_to_run):
            for horizontal_flip in [True, False]:
                for vertical_flip in [True, False]:
                    # for translation in well_spaced_translations():
                    for _ in range(num_translations):
                        try:
                            out = parse_run(
                                path_to_run=path_to_run,
                                path_to_images_dir=path_to_images_dir,
                                horizontal_flip=horizontal_flip,
                                vertical_flip=vertical_flip,
                                translation="random",
                                when=when,
                                mode=mode,
                                pixels=pixels,
                            )
                            metadata_listdict.append(out)
                        except:
                            print(f"# Something went wrong with {path_to_run}")

    metadata_df = pd.DataFrame(metadata_listdict)
    metadata_df.to_csv(output_metadata, index=False)


if __name__ == "__main__":
    fire.Fire()
