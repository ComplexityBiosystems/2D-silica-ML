# SilicaNets
Code to produce two-dimensional silica samples with pre-set ring-size distribution variance, in python 3. This code was used in 

```
ADD PUBLICATION
```


## Dependencies
+ python + tensorflow + standard packages
+ LAMMPS

## Installation
To install a fresh working python copy with tensorflow and all dependencies
with the correct versions, run
```bash
git clone ssh://git@ccnbnas.fisica.unimi.it:30001/francesc.font/silicanets.git
cd silicanets
conda create --name silicanets tensorflow pip matplotlib networkx pandas
conda activate silicanets
pip install temppathlib
pip install -e .
```


## Usage example
To create a two-dimensional silica configuration with pre-determined ring-size distribution $s^2=0.2$, with ten columns and 6 rows of rings, we can simply do as follows:

```python
# import dependecies
from silicanets.base import RingGraph

# set realistic physical parameters
radius_Si_O = 1.65
radius_Si_Si = 2 * radius_Si_O
dual_radius = 1.7320508075688776 * radius_Si_Si

# instantiate the graph
R = RingGraph(n_cols=10, n_rows=6, radius=1)
R.set_target(variance=0.2, alpha=0.33)

# perform the MCMC procedure to reach the desired ring size distro
R.mcmc(max_it=10000, temperature=1e-4)

# rescale to desired dual radius *only after* doing the full MCMC (dual relax works better with R=1)
R = R.rescale(factor=dual_radius)
```

at this point, `R` is a python object representing the two-dimensional silica sheet. To create the actual bilayer and export into a LAMMPS-compatible format, we do
```python
from pathlib import Path
from silicanets.io import get_lammps_string

# create the bilayer
G = inflate_silica_graph(S, thickness=radius_Si_Si)
lammps_string = get_lammps_string(G, three_dim=True)
Path("/path/to/output.lammps").write_text(lammps_string)
```




## Notes
For particular lammps file have to remove the 'ITEM: ATOMS id type xu yu zu c_peratom c_stratom[1]' and put 'ITEM: ATOMS id type xu yu zu c_peratom' in the io.py, because the lammps output are sometimes created with different columns.

## References
```
Morley, D.O., and Wilson, M. (2018). Controlling disorder in two-dimensional networks. J. Phys. Condens. Matter 30, 50LT02.  
Ebrahem, F., Bamer, F., and Markert, B. (2020). Vitreous 2D silica under tension: From brittle to ductile behaviour. Materials Science and Engineering: A 780, 139189.
```

## Implementation details
We follow the more detailed original manuscript (Morley, 2018). There are some important differences / missing details in (Ebrahem, 2020). Most importantly:
+ Elastic constant of dual bonds is not constant (should be equal to 1 according to Morley).
+ Definition of objective function $\chi$ differs slightly.
+ Morley gives details of parameters (MCMC temperature $10^{-4}$, $k=10$ in objective function).
  
### Calculation of $\alpha$
Following Morely, 2018, we compute alpha throught AW law. TODO finish explanation.
Basically, linear fit of
$$
L = R (1 - \alpha)
$$
