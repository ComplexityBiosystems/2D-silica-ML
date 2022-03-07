from pathlib import Path
from silicanets.io import get_lammps_string
from silicanets.utils import inflate_silica_graph
from silicanets.base import RingGraph
from pathlib import Path
import fire
from datetime import datetime


def main(
    path_to_output: str,
    ring_distro_variance: float = 1,
    n_cols: int = 16,
    n_rows: int = 18,
    radius_Si_O: float = 1.65,
    max_it: int = 10000,
):
    arguments = locals()
    # write to stdout for logs
    print("# Call log")
    print(datetime.now().isoformat())
    for k, v in arguments.items():
        print(k, v)
    print("")
    # make sure we can write output
    path_to_output: Path = Path(path_to_output)
    assert (
        not path_to_output.exists()
    ), "Output file already exists, refusing to overwrite."
    # create initial dual graph
    # dual radius is distance between nodes in dual graph
    # ring-to-ring distance measured from center of rings.
    radius_Si_Si = 2 * radius_Si_O
    dual_radius = 1.7320508075688776 * radius_Si_Si
    R = RingGraph(n_cols=n_cols, n_rows=n_rows, radius=1)
    
    #assert that type(ring_distro_variance)==float 
    if type(ring_distro_variance)==tuple:
        ring_distro_variance = float('.'.join(str(ele) for ele in ring_distro_variance))
        assert type(ring_distro_variance)==float 

    if type(ring_distro_variance)==str:
        ring_distro_variance = ring_distro_variance.replace('.','')
        ring_distro_variance = float(ring_distro_variance.replace(',','.'))
        assert type(ring_distro_variance)==float
    
    # bond switch for a while
    R.set_target(variance=ring_distro_variance, alpha=0.33)
    R.mcmc(max_it=max_it, temperature=1e-4)
    # rescale to desired dual radius *only after*
    # doing the full MCMC (dual relax works well with R=1)
    R = R.rescale(factor=dual_radius)
    # get actual SiO graph in 3D, write to file
    print("\n# Summary")
    R._stats()
    S = R.to_sio()
    G = inflate_silica_graph(S, thickness=radius_Si_Si)
    lammps_string = get_lammps_string(G, three_dim=True)
    path_to_output.write_text(lammps_string)


if __name__ == "__main__":
    fire.Fire(main)
