#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To Do
  - use symmetry of distance matrix to reduce memory/computation time
@author: Stefan Hiemer
"""

import numpy as np 
from postprocessing_functions import get_box_lengths
from numba import njit

def symmetry_funcs_general(coordinates, types, box, cutoff,
                          rad_center = [], rad_width = [],pbc = True):
    """
    This function makes sure, that only the same data types are being fed to 
    the underlying numba implementation. 
    
    coordinates: np.array of shape (number of atoms, number of dimensions) 
    types: np.array of shape (number of elements, number of dimensions)

    box: coordinates of box bounds as np.array of shape (2,ndim).
    cutoff: scalar, cutoff radius beyond which no calculations are done.
    rad_center: np.array
    rad_width: np.array
    pbc: periodic boundary conditions used for distance matrix
    return_params: boolean, wether to return the parameter grid of the symmetry
                   functions.
    generate_grid: boolean, wether to generate grid from symmetry parameter.
    
    Returns:
    radsym: np.array of shape (natoms, nradfeatures * ntypes) containing the
            Behler Parinello radial symmetry function features.
            
    """
    natoms, ndim = np.shape(coordinates)

    if isinstance(pbc, bool):
        pbc = np.tile(pbc,ndim)
    elif isinstance(pbc, np.ndarray):
        if np.shape(pbc)[0] == ndim:
            pass
        elif np.shape(pbc)[0] == 1 or len(np.shape(pbc)) == 0:
            pbc = np.tile(pbc,ndim)
        else:
            raise ValueError("Check passing valid periodic boundary condition information.")

    # make sure, that numba gets the correct datatypes
    coordinates = np.float64(coordinates)
    types = np.int64(types)

    box_length, tilt_factors = get_box_lengths(box,return_tilts=True)
    box_length = np.float64(box_length)
    tilt_factors = np.float64(tilt_factors)

    cutoff = np.float64(cutoff)
    rad_center = np.float64(rad_center)
    rad_width = np.float64(rad_width)

    rad_sym = _symmetry_funcs_general(coordinates, types, box, tilt_factors,
                                      cutoff, rad_center, rad_width,pbc)
    return rad_sym

@njit(nopython=True)
def _symmetry_funcs_general(coordinates, types, box, tilt_factors, cutoff,
                          rad_center, rad_width,pbc):
    """
    Simple numba implementation in order to calculate the radial distribution 
    functions.

    coordinates: np.array of shape (number of atoms, number of dimensions) 
    types: np.array of shape (number of elements, number of dimensions)

    box: coordinates of box bounds as np.array of shape (2,ndim).
    cutoff: scalar, cutoff radius beyond which no calculations are done.
    rad_center: np.array
    rad_width: np.array
    pbc: np.array of shape (3,) periodic boundary conditions used for the 
         calculation of the distance matrix.
    return_params: boolean, wether to return the parameter grid of the symmetry
                   functions.
    generate_grid: boolean, wether to generate grid from symmetry parameter.
    
    Returns:
    radsym: np.array of shape (natoms, nradfeatures * ntypes) containing the
            Behler Parinello radial symmetry function features.

    """


    print("Calculating Radial Symmetry Function!")

    natoms, ndim = np.shape(coordinates)
    type_labels = np.sort(np.unique(types))
    ntypes = np.shape(type_labels)[0]
    n_rad_features = np.shape(rad_center)[0]

    # calculate box lengths for periodic boundary conditions
    if pbc.any():
        box_length = box[:,1] - box[:,0]

    # initiate container with a silly trick to allow numba to infer the type
    # of the list elements.
    radsym = np.zeros((natoms,n_rad_features*ntypes),
                      dtype=np.float64)

    # iterate over atoms
    for i in np.arange(natoms):
        r_i = coordinates[i]

        for j in np.arange(int(i+1),natoms):

            r_vec = coordinates[j] - r_i

            # apply periodic boundary conditions
            if pbc.any():
                half = box_length/2
                mask1 = (r_vec < -half)
                mask2 = (r_vec > half)
                # conventional for orthogonal boxes
                r_vec += mask1 * pbc * box_length
                r_vec -= mask2 * pbc * box_length

                # z
                if pbc[2] and mask1[2]:
                    r_vec[0] += tilt_factors[1] # xz
                    r_vec[1] += tilt_factors[2] # yz
                elif pbc[2] and mask2[2]:
                    r_vec[0] -= tilt_factors[1] # xz
                    r_vec[1] -= tilt_factors[2] # yz

                # y
                if pbc[1] and mask1[1]:
                    r_vec[0] +=  tilt_factors[0] # xy
                elif pbc[1] and mask2[1]:
                    r_vec[0] -=  tilt_factors[0] # xy

            r = np.linalg.norm(r_vec)

            # apply cutoff
            if r > cutoff:
                continue
            else:

                # calculate radial symmetry function
                rad = np.exp(-(r - rad_center)**2 /rad_width**2)
                # choose columns where to save the information according to type
                position = n_rad_features * np.where(type_labels==types[j])[0][0]

                # append to radial symmetry function
                radsym[i,position:position+n_rad_features] += rad

                # choose columns where to save the information according to type
                position = n_rad_features * np.where(type_labels==types[i])[0][0]

                # append to radial symmetry function
                radsym[j,position:position+n_rad_features] += rad


    return radsym 

def symmetry_funcs_binary(coordinates, types, box, cutoff,
                          rad_center = [], rad_width = [], pbc = True,
                          return_params = False, generate_grid = True):
    """
    This function is for test reasons only as it is much too slow to be used for
    anything else.

    coordinates: np.array of shape (number of atoms, number of dimensions)
    types: np.array of shape (number of elements, number of dimensions)

    box: coordinates of box bounds as np.array of shape (2,ndim).
    cutoff: scalar, cutoff radius beyond which no calculations are done.
    rad_center: np.array
    rad_width: np.array 
    pbc: periodic boundary conditions used for distance matrix
    return_params: boolean, wether to return the parameter grid of the symmetry
                   functions.
    generate_grid: boolean, wether to generate grid from symmetry parameter.
    
    
    Returns:
    radsym: np.array of shape (natoms, nradfeatures * ntypes) containing the
            Behler Parinello radial symmetry function features. 

    """
    print("Calculating Radial Symmetry Function!")

    # find parameters of system
    natoms, ndim = np.shape(coordinates)
    type_labels = np.sort(np.unique(types))

    # check that the system is indeed binary
    assert(np.size(type_labels) == 2)  


    # distance matrix along each coordinate direction of shape
    # (natoms,natoms,ndim)
    distance_vec_matrix = coordinates[np.newaxis, :, :] - \
                          coordinates[:, np.newaxis, :]

    # periodic boundary conditions (should be adapted for mixed boundaries)
    if pbc:
        # parameters needed for pbc
        box_length = box[:,1] - box[:,0]
        box_length = box_length[np.newaxis, np.newaxis, :]
        # apply periodic boundary conditions
        distance_vec_matrix += (distance_vec_matrix <  box_length/2)\
                               .astype(np.float32) * box_length
        distance_vec_matrix -= (distance_vec_matrix > box_length/2)\
                               .astype(np.float32) * box_length

    # calculates final global distance matrix
    distance_matrix = np.linalg.norm(distance_vec_matrix, axis=-1)

    # apply cutoff and kick out diagonal
    general_mask = (distance_matrix < cutoff) & ~np.eye(natoms, dtype=bool)

    # create mask for particles of same type
    type_mask = (types[:,None] == types[None,:])

    # radial distribution function
    radsym = np.exp(-(distance_matrix[:,None,:] - rad_center[None,:,None])**2 \
             /rad_width[None,:,None]**2)
    radsym = np.concatenate(
             (np.sum((general_mask & type_mask)[:,None,:] * radsym, axis = -1),
             np.sum((general_mask & ~type_mask)[:,None,:] * radsym, axis = -1)),
             axis = -1) 
    
    if return_params:
        params = {'radial width': rad_width,
                  'radial center': rad_center}
        return radsym, params
    else:
        return radsym

def check(a,b):
    """
    Function used in tests to find out wether to arrays are approximately the
    same in all elements.

    a: np.array
    b: np.array
    """
    return np.all(np.isclose(a,b))

def test_binary_rad():
    """
    Test the radial symmetry function for a toy example of 4 particles.
    Currently the tests are hard coded. Maybe in the future build it in a more
    dynamic sense to enable change of tests.
    """

    _rad_center = np.linspace(1,2,2),
    _rad_width = np.linspace(1,2,2),

    _rad_center, _rad_width = np.meshgrid(_rad_center, _rad_width)
    _rad_center, _rad_width = _rad_center.flatten(), _rad_width.flatten()

    sym, params = symmetry_funcs_binary(coordinates = np.array([[1.,0,0],
                                                                [0,1.,0],
                                                                [0,0,1.],
                                                                [1.,1.,0]]), 
                                        types = np.array([1,2,1,2]),
                                        box = np.array([[-6,6.],[-6,6.],[-6,6.]]),
                                        cutoff = 10,
                                        rad_center = _rad_center,
                                        rad_width = _rad_width,
                                        pbc = False, return_params = True)




    # check if shape of radial symmetry function is correct
    assert(sym.shape == (4,8))

    # check if radial symmetry function for particle 1 distances is correct
    # and at the correct place
    assert(check(sym[0,:4] , np.exp(-(np.sqrt(2) - _rad_center)**2 /_rad_width**2)))
    assert(check(sym[0,4:] , (np.exp(-(np.sqrt(2) - _rad_center)**2 /_rad_width**2) +\
           np.exp(-(1 - _rad_center)**2 /_rad_width**2))))

    # check if radial symmetry function for particle 2 distances is correct
    # and at the correct place
    assert(check(sym[1,:4] , np.exp(-(np.sqrt(1) - _rad_center)**2 /_rad_width**2)))
    assert(check(sym[1,4:] , (np.exp(-(np.sqrt(2) - _rad_center)**2 /_rad_width**2) +\
          np.exp(-(np.sqrt(2) - _rad_center)**2 /_rad_width**2))))

    return

def test_binary_rad_new():
    """
    Test the radial symmetry function for a toy example of 4 particles.
    Currently the tests are hard coded. Maybe in the future build it in a more
    dynamic sense to enable change of tests.
    """

    _rad_center = np.linspace(1,2,2),
    _rad_width = np.linspace(1,2,2),

    _rad_center, _rad_width = np.meshgrid(_rad_center, _rad_width)
    _rad_center, _rad_width = _rad_center.flatten(), _rad_width.flatten()

    sym, params = symmetry_funcs_binary(coordinates = np.array([[1.,0,0],
                                                                [0,1.,0],
                                                                [0,0,1.],
                                                                [1.,1.,0]]), 
                                        types = np.array([1,2,1,2]),
                                        box = np.array([[-6,6.],[-6,6.],[-6,6.]]),
                                        cutoff = 10,
                                        rad_center = _rad_center,
                                        rad_width = _rad_width,
                                        pbc = False, return_params = True)




    # check if shape of radial symmetry function is correct
    assert(sym.shape == (4,8))

    # check if radial symmetry function for particle 1 distances is correct
    # and at the correct place
    assert(check(sym[0,:4] , np.exp(-(np.sqrt(2) - _rad_center)**2 /_rad_width**2)))
    assert(check(sym[0,4:] , (np.exp(-(np.sqrt(2) - _rad_center)**2 /_rad_width**2) +\
           np.exp(-(1 - _rad_center)**2 /_rad_width**2))))

    # check if radial symmetry function for particle 2 distances is correct
    # and at the correct place
    assert(check(sym[1,:4] , np.exp(-(np.sqrt(1) - _rad_center)**2 /_rad_width**2)))
    assert(check(sym[1,4:] , (np.exp(-(np.sqrt(2) - _rad_center)**2 /_rad_width**2) +\
          np.exp(-(np.sqrt(2) - _rad_center)**2 /_rad_width**2))))

    return

def test_general(natoms = 100, ndim = 3, seed = 0):
    """
    Test the radial symmetry function for a toy example of 4 particles.
    Currently the tests are hard coded. Maybe in the future build it in a more
    dynamic sense to enable change of tests.
    """
    
    # draw random seed
    np.random.seed(seed)
    
    # initialize hyperparameters of the raidal symmetry functions
    _rad_center = np.linspace(1,2,2),
    _rad_width = np.linspace(1,2,2),
    
    _rad_center, _rad_width = np.meshgrid(_rad_center, _rad_width)
    _rad_center, _rad_width = _rad_center.flatten(), _rad_width.flatten()
    
    # create the system with random coordinates, predefined box parameters,
    # random binary atom types, a predefined cutoff and periodic boundary 
    # conditions
    coordinates = np.random.rand(natoms,ndim)
    box = np.array([[0,1],[0,1],[0,1]])
    types = np.random.randint(0,2,(natoms))
    cutoff = 0.25
    pbc = np.array([True])
    
    # calculate symmetry functions
    sym = symmetry_funcs_binary(coordinates = coordinates,
                                types = types,
                                box = box,
                                cutoff = cutoff,
                                rad_center = _rad_center,
                                rad_width = _rad_width,
                                pbc = pbc, return_params = False)

    _sym = symmetry_funcs_general(coordinates = coordinates,
                                  types = types,
                                  box = box,
                                  cutoff = cutoff,
                                  rad_center = _rad_center,
                                  rad_width = _rad_width,
                                  pbc = pbc)
    
    # This is done as both functions follow different sorting conventions.
    # It does not matter which sorting convention is being followed as long
    # as only one convention is being used. As I want to compare the slow 
    # function symmetry_funcs_binary with its faster version I need to adapt 
    # the sorting in order to compare them.
    mask = types == 1
    sym[mask] = np.roll(sym[mask],np.shape(_rad_center)[0],axis=1)

    assert np.allclose(_sym, sym) , "General version of binary symmetry function test failed."

    return

if __name__ == '__main__':

    test_binary_rad_new()
    test_general()
