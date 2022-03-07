#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Stefan Hiemer
"""

from postprocessing_functions import read_dump
import numpy as np

def box_affine_transformation(coordinates, box,startindex = None):
    """
    coordinates: np.array of shape either (nsnaps,natoms,ndim) or (natoms,ndim)
    box: np.array of shape (nsnaps,natoms,ndim)
    """
    if len(coordinates.shape) == 3:
        if not startindex:
            coordinates = coordinates[0]
        else:
            coordinates = coordinates[startindex]

    nsnap, ndim, nboxparams = np.shape(box)

    # orthogonal box
    if (nboxparams == 2 and ndim == 3) or \
        (nboxparams == 2 and ndim == 2):

        box_lengths = box[:,:,1] - box[:,:,0]
        affine_elongations = box_lengths[1:]/box_lengths[0:1]

        xlo = box[:,0,0]
        ylo = box[:,1,0]
        zlo = box[:,2,0]

        def_grad = np.zeros((int(nsnap-1),ndim,ndim))
        def_grad[:,0,0] = affine_elongations[:,0]
        def_grad[:,1,1] = affine_elongations[:,1]
        def_grad[:,2,2] = affine_elongations[:,2]

        del affine_elongations

    # 3D triclinic box
    elif nboxparams == 3 and ndim == 3:
        # calculate box parameters (LAMMPS manual triclinic box)
        xy = box[:,0,2]
        xz = box[:,1,2]
        yz = box[:,2,2]
        xlo = box[:,0,0] - np.min([np.zeros(nsnap),xy,xz,xy+xz], axis=0)
        xhi = box[:,0,1] - np.max([np.zeros(nsnap),xy,xz,xy+xz], axis=0)
        ylo = box[:,1,0] - np.min([np.zeros(nsnap),yz], axis=0)
        yhi = box[:,1,1] - np.max([np.zeros(nsnap),yz], axis=0)
        zlo = box[:,2,0]
        zhi = box[:,2,1]

        box_lengths = np.column_stack([xhi-xlo, yhi-ylo, zhi-zlo])
        affine_elongations = box_lengths[1:]/box_lengths[0:1]

        def_grad = np.zeros((int(nsnap-1),ndim,ndim))
        def_grad[:,0,0] = affine_elongations[:,0]
        def_grad[:,1,1] = affine_elongations[:,1]
        def_grad[:,2,2] = affine_elongations[:,2]
        def_grad[:,0,1] = (xy[1:]-xy[0])/box_lengths[0,1] #xy
        #def_grad[:,1,0] = def_grad[:,0,1] #xy
        def_grad[:,0,2] = (xz[1:]-xz[0])/box_lengths[0,2] #xz
        #def_grad[:,2,0] = def_grad[:,0,2] #xz
        def_grad[:,1,2] = (yz[1:]-yz[0])/box_lengths[0,2] #yz
        #def_grad[:,2,1] = def_grad[:,1,2] #yz

        del affine_elongations

    # transform to coordinates system with origin (xmin,ymin,zmin) at (0,0,0)
    coordinates = coordinates[None,:,:] - \
                           np.column_stack([xlo, ylo, zlo])[1:,None,:]

    affine_transformed = np.matmul(coordinates, np.transpose(def_grad,
                                   axes=[0,2,1]))
    # transform back to old coordinates
    affine_transformed = affine_transformed + np.column_stack([xlo, ylo, zlo])[1:,None,:]

    return affine_transformed

def test_box_affine_transformation():
    """
    Check my implementation versus a lammps script
    """
    dump = read_dump("./affine-transformation-test/SiO2-BKS_554195833-975147688.data_aqs-unishear-+xy-strainincrement0.0001.dump",
              snapshot_indices = [0,1,2,4], sort = True)

    labels = dump["labels"]

    coordinates =  dump["snapshots"][:,:,int(labels.index("x")):\
                                        int(labels.index("x")+3)]

    affine_displacements = box_affine_transformation(coordinates[0,:,:],
                                                     dump["box dimensions"])


    assert np.all(np.isclose(dump["snapshots"][1:,:,int(labels.index("x")):int(labels.index("x")+3)],
                             affine_displacements[:,:,:])), \
                             "Affine transformation test does not work"
    return

if __name__=="__main__":

    test_box_affine_transformation()
