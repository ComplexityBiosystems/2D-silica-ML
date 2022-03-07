#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Stefan Hiemer
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import sys
from functools import partial

def _sort_id(snapshot, col_nr):
    """
    Return a single snapshot sorted according to atom id.

    snapshot: np.array (natoms, natomprops), snapshot with lammps dump atomic
              properties.
    col_nr: int, column which includes the atom ids.
    """

    indices = np.argsort(snapshot[:,col_nr])
    snapshot = snapshot[indices]

    return snapshot

def read_dump(file, snapshot_indices = None, sort = False):

    with open (file,"r") as text:


        box_options = []

        timesteps = []
        atom_numbers = []
        labels = []
        box_dim = []
        snapshots = []
        #read start
        snapshot = []
        """
        This loop reads the per-atom data with a while loop for every loop not
        containing ITEM. For all global information (timestep, box, etc.) if
        forks are used which manually jump over the lines containing their
        input.
        """
        if not snapshot_indices:
            for line in text:
                #remove multiple whitespaces and spaces at beginning and end
                line = ' '.join(line.strip().split())
                print(line)

                while "ITEM" not in line and len(line)!=0:
                    snapshot = np.vstack((snapshot,line.strip().split(' ')))
                    line = ' '.join(text.readline().strip().split())

                if "ITEM: TIMESTEP" in line:
                    timesteps.append(int(text.readline().strip()))
                elif "ITEM: NUMBER OF ATOMS" in line:
                    atom_numbers.append(int(text.readline().strip()))
                elif "ITEM: BOX BOUNDS " in line:
                    # special case if box is orthogonal in 2d or 3d
                    if len(line.split(' ')) == 5 or len(line.split(' ')) == 6:
                        box_options.append(line.split(' ')[3:])
                    # special case if box is 2d and non orthogonal
                    elif len(line.split(' ')) == 7:
                        box_options.append(line.split(' ')[5:])
                    # special case if box is 3d and non orthogonal
                    elif len(line.split(' ')) == 9:
                        box_options.append(line.split(' ')[6:])
                    # 3d system
                    if len(box_options[-1]) == 3:
                        box_dim.append([text.readline().strip().split(' '),
                                        text.readline().strip().split(' '),
                                        text.readline().strip().split(' ')])
                    # 2d system
                    elif len(box_options[-1]) == 2:
                        box_dim.append([text.readline().strip().split(' '),
                                        text.readline().strip().split(' ')])
                elif "ITEM: ATOMS " in line and len(labels) != 0:
                    #append previous snapshot to the collection
                    snapshots.append(snapshot)
                    #read atom coordinates and first line of snapshots to get the
                    #number of columns for the np array
                    line = ' '.join(text.readline().strip().split())
                    snapshot = np.array(line.split(' '))
                elif "ITEM: ATOMS " in line and len(labels) == 0:
                    #coordinates_start.append(int(row_index+1))
                    labels = line.split(' ')[2:]
                    #read atom coordinates and first line of snapshots to get the
                    #number of columns for the np array
                    line = ' '.join(text.readline().strip().split())
                    snapshot = np.array(line.split(' '))

            #append last snapshot
            snapshots.append(snapshot)
        else:
            snapshotcount = 0
            read = False
            for line in text:

                #remove multiple whitespaces and spaces at beginning and end
                line = ' '.join(line.strip().split())

                while "ITEM" not in line and len(line)!=0 and read:
                    snapshot = np.vstack((snapshot,line.strip().split(' ')))
                    line = ' '.join(text.readline().strip().split())

                # update snapshotcount
                if "ITEM: TIMESTEP" in line:
                    read = snapshotcount in snapshot_indices
                    snapshotcount = snapshotcount + 1

                if read:

                    if "ITEM: TIMESTEP" in line:
                        timesteps.append(int(text.readline().strip()))
                    elif "ITEM: NUMBER OF ATOMS" in line:
                        atom_numbers.append(int(text.readline().strip()))
                    elif "ITEM: BOX BOUNDS " in line:
                        # special case if box is orthogonal in 2d or 3d
                        if len(line.split(' ')) == 5 or len(line.split(' ')) == 6:
                            box_options.append(line.split(' ')[3:])
                        # special case if box is 2d and non orthogonal
                        elif len(line.split(' ')) == 7:
                            box_options.append(line.split(' ')[5:])
                        # special case if box is 3d and non orthogonal
                        elif len(line.split(' ')) == 9:
                            box_options.append(line.split(' ')[6:])
                        # 3d system
                        if len(box_options[-1]) == 3:
                            box_dim.append([text.readline().strip().split(' '),
                                            text.readline().strip().split(' '),
                                            text.readline().strip().split(' ')])
                        # 2d system
                        elif len(box_options[-1]) == 2:
                            box_dim.append([text.readline().strip().split(' '),
                                            text.readline().strip().split(' ')])
                    elif "ITEM: ATOMS " in line and len(labels) != 0:
                        #append previous snapshot to the collection
                        snapshots.append(snapshot)
                        #read atom coordinates and first line of snapshots to get the
                        #number of columns for the np array
                        line = ' '.join(text.readline().strip().split())
                        snapshot = np.array(line.split(' '))
                    elif "ITEM: ATOMS " in line and len(labels) == 0:
                        #coordinates_start.append(int(row_index+1))
                        labels = line.split(' ')[2:]
                        #read atom coordinates and first line of snapshots to get the
                        #number of columns for the np array
                        line = ' '.join(text.readline().strip().split())
                        snapshot = np.array(line.split(' '))

            #append last snapshot
            snapshots.append(snapshot)

    # sort according to atom id
    if sort:
        # find out column
        print(labels)
        col_nr = labels.index('id')

        sort_id = partial(_sort_id, col_nr = col_nr)

        snapshots = list(map(sort_id, snapshots))

    #convert lists to np array
    box_options = np.array(box_options)
    box_dim =  np.array(box_dim).astype('float')
    timesteps = np.array(timesteps).astype('int')
    atom_numbers = np.array(atom_numbers).astype('int')
    snapshots = np.array(snapshots).astype('float')

    return {"labels": labels, "time": timesteps, "snapshots": snapshots,
            "box dimensions": box_dim, "boundary conditions": box_options,
            "number of atoms": atom_numbers}

def read_rdf_output(file):
    """
    Read output of the fix ave/time command for the radial distribution function
    (rdf) from LAMMPS. The full lammps command could be:
    fix             1 all ave/time 1 1 100 c_rdf[*] file test.rdf mode vector
    The rdf is calculated by counting the number of atoms within a distance
    interval (bin) around each atom and for normalization divide it by the
    particle (not mass!) density.

    file: string, filename
    """

    # find out number of bins
    with open(file,'r') as f:
        for line in f:
            if '# Time' in line:
                continue
            elif '# Row' in line:
                number_bins = int(f.readline().strip().split(' ')[1])
                break
    # collect rdf
    with open(file,'r') as f:
        timesteps = []
        bin_values = []
        for line in f:

            if len(line.strip().split(' ')) == 2:
                timesteps.append(int(line.strip().split(' ')[0]))
            elif '#' in line:
                continue
            elif not line.strip():
                break
            else:
                rdf_snapshot = []
                rdf_snapshot.append(np.array(line.strip().split(' ')[1:]).astype(float))
                for i in range(int(number_bins-1)):
                    rdf_snapshot.append(np.array(f.readline().strip().split(' ')[1:]).astype(float))
                bin_values.append(np.array(rdf_snapshot))

    timesteps = np.array(timesteps)
    bin_values = np.array(bin_values)

    return timesteps, bin_values

def get_absolute_coordinates(coordinates, periodic_flags, box, test = False):
    """
    coordinates: np.array, shape (nsnapshots, natoms, ndim)
    periodic_flags: np.array, shape (nsnapshots, natoms, ndim)
    box np.array, shape (nsnapshots,ndim,nboxparam)

    """

    nsnap = box.shape[0]

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
    if test:
        print("xlo: ", xlo, "xhi: ", xhi,
              "ylo: ", ylo, "yhi: ", yhi,
              "zlo: ", zlo, "zhi: ", zhi,
              "xy: ", xy, "xz: ", xz,
              "yz: ", yz)

    # case where periodic flags are provided
    if type(periodic_flags) is np.ndarray:
        periodic_flags = periodic_flags - periodic_flags[0][None,:,:]
        absolute_coordinates = coordinates + periodic_flags * box_lengths[:,None,:]
        return absolute_coordinates
    else:
        sys.exit()

def test_get_absolute_coordinates():

    _dump = read_dump("absolute-coordinates-test.dump")
    labels = _dump["labels"]
    abs_coordinates = get_absolute_coordinates(
                              _dump["snapshots"][:,:,int(labels.index("x")):\
                                        int(labels.index("x")+3)],
                              _dump["snapshots"][:,:,int(labels.index("ix")):\
                                        int(labels.index("ix")+3)],
                              _dump['box dimensions'])

    assert np.allclose(abs_coordinates[1],
                       np.array([[-7.9, 1.5, 2.5],
                                 [-4.5, 11.1, -2.5],
                                 [2.5, 0.5, 1.5,],
                                 [-2.5, -0.5, -14.3]])),\
            "3D: Function test_get_absolute_coordinates failed"

def get_box_lengths(box,return_tilts=False):
    """
    Converts the box options given by the LAMMPS dump files. While in the
    case of orthogonal boxes, this is trivial, but in triclinic box one has to
    follow the conventions.

    box: np.array shape (nsnapshots,ndim,nboxparam) or (2,ndim), coordinates of
         box boundaries and box tilts as np.array. If shape (2,ndim) constant box volume for
         all snapshots is assumed.
    """

    # check if its just a single snapshot
    size = len(np.shape(box))

    if size == 2:
        box = np.expand_dims(box,axis=0)

    nsnap, ndim, nboxparams = np.shape(box)

    # orthogonal box
    if (nboxparams == 2 and ndim == 3) or \
        (nboxparams == 2 and ndim == 2):

        box_lengths = box[:,:,1] - box[:,:,0]

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

    # 2D triclinic box
    elif nboxparams == 3 and ndim == 2:
        print("Not done yet!")
        sys.exit()

    if return_tilts:
        # case if box orthogonal and no tilt factors were written
        if nboxparams == 2:
            box = np.append(box,np.zeros((nsnap,ndim,1)),axis=-1)
        if size == 2:
            return box_lengths[0],box[0,:,2]
        else:
            return box_lengths,box[:,:,2]
    else:
        if size == 2:
            return box_lengths[0]
        else:
            return box_lengths


if __name__ == "__main__":
    pass
