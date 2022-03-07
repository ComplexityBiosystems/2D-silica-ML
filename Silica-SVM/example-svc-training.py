#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Stefan Hiemer
"""

from postprocessing_functions import *

import glob
from symmetry_functions import symmetry_funcs_general
from transformation import box_affine_transformation
import numpy as np
import os
import re

data_location="/FASTTEMP/test/shiemer/glass-ml/molecular-dynamics/watanabe/aqs/var0.2"
home=os.getcwd()
strain = 0.05
cutoff = 10.0
rad_min = 0.2
rad_stop = 7.6
interval = 0.2

rad_center = np.arange(rad_min,rad_stop,interval)
rad_width = np.array([0.1])
rad_name = "-"+str(rad_min)+'-'+str(rad_stop)+'-'+str(interval)+'-'+str(cutoff)

def data_collection(start=0,
                    end=10):
    """
    data_location: string, path where the atomic data has been stored.
    start: int, index for the directories from which to start collecting data.
    end: int, index for the directories from which to stop collecting data.
    """

    os.chdir(data_location)

    # hyperparameter
    natoms = 3456

    # find directories containing data
    directories = glob.glob("*run*")

    # sort files
    directories = sorted(directories)

    # create dictionary to collect data
    feat = {"initial":[],
            "affine":[],
            "broken":[]}

    for directory in directories[start:end]:

        os.chdir(directory)

        # check wether calculation was successful finished
        if not os.path.isfile("sio2_3_bondbreak_1.is_broken"):
            os.chdir("..")
            continue

        # check wether only three atoms are there
        with open("bondbreak.dat","r") as file:
            file.readline()
            data = file.readline()
        if len(data.split(",")) != 6:
            os.chdir("..")
            continue

        # get seed from directory name
        seed = np.array([int(directory.split("_")[1])]).reshape((1,1))

        # read dump files
        dump = read_dump("sio2_1_initial.dump",
                         snapshot_indices = [0],
                         sort = True)

        # get column ids
        x = dump["labels"].index('xu')
        z = int(dump["labels"].index('zu') + 1)
        type_id = dump["labels"].index('type')

        # wrap coordinates along periodic boundaries
        box_lengths=dump["box dimensions"][:,:,1]-dump["box dimensions"][:,:,0]
        mask = dump["snapshots"][:,:,x:z] > dump["box dimensions"][:,None,:,1]
        dump["snapshots"][:,:,x:z] = dump["snapshots"][:,:,x:z] - mask * box_lengths[:,None,:]
        mask = dump["snapshots"][:,:,x:z] < dump["box dimensions"][:,None,:,0]
        dump["snapshots"][:,:,x:z] = dump["snapshots"][:,:,x:z] + mask * box_lengths[:,None,:]

        # calculate features for initial snapshot
        rad = symmetry_funcs_general(coordinates = dump["snapshots"][0,:,x:z],
                                     types = dump["snapshots"][0,:,type_id],
                                     box = dump["box dimensions"][0,:,:],
                                     cutoff = cutoff,
                                     rad_center = rad_center,
                                     rad_width = rad_width,
                                     pbc = True)

        # read information for the first bond break
        broken = get_broken()

        print("Radial Symmetry finished:",seed)

        # construct an affine
        box = np.copy(dump["box dimensions"])
        box[0,0,1] = (1+strain)*box[0,0,1]

        # apply affine transformation to initial snapshot
        aff = box_affine_transformation(dump["snapshots"][0,:,x:z],
                             np.append(dump["box dimensions"],
                             box,
                             axis=0))

        # calculate features for transformed snapshots
        rad_shear = symmetry_funcs_general(
                                       coordinates = aff[0],
                                       types = dump["snapshots"][0,:,type_id],
                                       box = box[0,:,:],
                                       cutoff = cutoff,
                                       rad_center = rad_center,
                                       rad_width = rad_width,
                                       pbc = True)

        print("Shear finished:",seed)

        feat["initial"].append(rad)
        feat["affine"].append(rad_shear)
        feat["broken"].append(broken)

        os.chdir("..")

    # turn data to np arrays for better handling
    feat["initial"] = np.stack(feat["initial"])
    feat["affine"] = np.stack(feat["affine"])
    feat["broken"] = np.stack(feat["broken"])

    return feat

def get_broken():
    """
    Reads the postprocessing file which contains infromation about the
    atoms involved in the first bond breaking. Returns a boolean np.array of
    shape (natoms) where the two atoms which break the first bonds are true,
    everything else is False.
    """

    # read file which contains information about atoms involved in the first
    # bond break
    with open("bondbreak.dat",'r') as file:
        # drop first line
        file.readline()
        # second line contains the data
        data = file.readline()

    data = re.split(",|=|:",data)

    # get atom id of deformed oxygen
    atom_id_O = int(data[1])

    # get bond lengths
    bond_length1 = float(data[-5].strip())
    bond_length2 = float(data[-1][:-2])

    # determine the silicon atom which is the partner in the broken bond
    if bond_length1 > bond_length2:
        atom_id_Si = int(data[-7])
    else:
        atom_id_Si = int(data[-3])

    # read dump file of initial configuration
    dump = read_dump("sio2_1_initial.dump",
                     snapshot_indices = [0],
                     sort = True)

    # get the sorted atom indices from file. please note, that due to some
    # fault on my side I sort the indices in string format, so they are not
    # sorted according to size, but to string ordering. If you are unsure what
    # this means simply convert np.arange(1000) to str, sort and see what happens.
    id = dump["labels"].index('id')

    # find the atom ids of the atoms which have broken a bond
    ids = np.append(np.where(np.isclose(dump["snapshots"][0,:,id],atom_id_O))[0],
                    np.where(np.isclose(dump["snapshots"][0,:,id],atom_id_Si))[0])
    deformed = np.zeros(3456,dtype="bool")
    deformed[ids] = True

    return deformed

def train_test_split(data,
                     affine_only=False,
                     maxsamples=None,
                     balanced=True,
                     max_atoms=10000,
                     seed=1):
    """
    data: np.array of shape (nsampes,natoms,dimensions)
    affine_only: bool, if true only symmetry functions from the affinely
                 transformed configuration are used.
    maxsamples: int, maximum number of simulations from which data is generated
    balanced: bool, if true subsampling as described in the publication is done
    max_atoms: integer, maximum number of atoms used to train the SVC.
    seed: integer, random seed used for the subsampling.
    """
    nsamples = data["initial"].shape[0]

    # hard coded 80/20 split
    if maxsamples:
        if maxsamples > nsamples:
            ind = int(nsamples * 0.8)
            maxsamples = nsamples
        else:
            ind = int(maxsamples * 0.8)
    else:
        ind = int(nsamples*0.8)

    # combine the initial and affine structure
    if affine_only:
        xtrain = data["affine"][:ind]
        xtest = data["affine"][ind:]

    else:
        x = np.concatenate((data["initial"],
                            data["affine"]),
                            axis=-1)

        xtrain = x[:ind]
        xtest = x[ind:]

    # split targets samplewise
    xtrain, xtest = np.vstack(xtrain),np.vstack(xtest)
    ytrain = np.hstack(data["broken"][:ind])
    ytest = np.hstack(data["broken"][ind:])

    # subsampling
    if balanced:

        # intialize random seed generator
        np.random.seed(seed)

        # get indices of deformed particle in the training set
        train_index = np.nonzero(ytrain)[0]
        nondeformed = np.delete(np.arange(np.shape(ytrain)[0]),
                                                   train_index)
        if np.shape(train_index)[0] > int(max_atoms/2):
            train_index = np.random.choice(train_index,
                                           int(max_atoms/2))

        train_index = np.concatenate([np.random.choice(nondeformed,
                                      np.shape(train_index)[0]),
                                      train_index])

        test_index = np.nonzero(ytest)[0]
        test_index = np.concatenate([np.random.choice(np.delete(
                                     np.arange(np.size(ytest)),
                                     test_index),
                                     np.size(test_index)),
                                     test_index])

        xtrain = xtrain[train_index,:]
        xtest = xtest[test_index,:]
        ytrain = ytrain[train_index]
        ytest = ytest[test_index]

    return xtrain, xtest, ytrain, ytest

def train():
    """
    This function optimizes the support vector classifier (SVC) for a 
    wide choice of hyperparameters of the SVC. 
    """
    
    # calculates the symmetry functions and draws the targets
    data = data_collection(start=0,
                           end=10)
    
    # performs the train-test split as discussed in the paper.
    xtrain,xtest,ytrain,ytest = train_test_split(data,
                                                 affine_only=False,
                                                 balanced=True)

    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix

    gamma = np.append(np.column_stack((np.logspace(-5,4,10),
                                       5*np.logspace(-5,4,10))).flatten(),
                      10**5)

    svc_grid = GridSearchCV(
                 SVC(),
                 param_grid =[{"kernel": ["linear"],
                               "shrinking": [True],
                               "probability": [True],
                               "tol": [0.001],
                               "cache_size": [200],
                               "class_weight": ['balanced'],
                               "verbose": [False],
                               "max_iter": [-1],
                               "decision_function_shape": ['ovr'],
                               "break_ties": [False],
                               "random_state": [1],
                               "C": [0.01, 0.1, 0.5, 1.0, 2.0, 10, 100]},
                              {"kernel": ["rbf"],
                               "shrinking": [True],
                               "probability": [True],
                               "tol": [0.001],
                               "cache_size": [200],
                               "class_weight": ['balanced'],
                               "verbose": [False],
                               "max_iter": [-1],
                               "decision_function_shape": ['ovr'],
                               "break_ties": [False],
                               "random_state": [1],
                               "gamma": gamma,
                               "C": [0.01, 0.1, 0.5, 1.0, 2.0, 10, 100]}],
                 cv = 5,
                 n_jobs = -1,
                 verbose = 10,
                 scoring = {"accuracy": "accuracy",
                            "recall": 'recall',
                            "roc_auc": 'roc_auc',
                            "precision": "precision"},
                 refit = 'accuracy',
                 return_train_score = True)

    # perform gridsearch
    svc_grid = svc_grid.fit(xtrain, ytrain)

    #
    ypred = svc_grid.best_estimator_.predict(xtest)

    # create confusion matrix
    cmd = ConfusionMatrixDisplay(confusion_matrix(ytest, ypred,
                                                  normalize = 'all'))
    cmd.plot()
    plt.show()

    create_ovito_snapshot(nsnaps=1,svc=svc_grid.best_estimator_)

    return

def create_ovito_snapshot(nsnaps,
                          svc,
                          affine_only=False):

    # find directories containing data
    directories = glob.glob("*run*")

    for directory in directories[-nsnaps:]:

        os.chdir(directory)

        # check wether calculation was successful finished
        if not os.path.isfile("sio2_3_bondbreak_1.is_broken"):
            os.chdir("..")
            continue

        # check wether only three atoms are there
        with open("bondbreak.dat","r") as file:
            file.readline()
            data = file.readline()
        if len(data.split(",")) != 6:
            os.chdir("..")
            continue

        # get seeds
        number = directory.split("_")[1]

        # convert seeds to integer as h5py is too stupid to handle strings easy
        seed = np.array([int(number)]).reshape((1,1))

        # read dump files
        dump_init = read_dump("sio2_1_initial.dump",
                              snapshot_indices = [0],
                              sort = True)
        dump_broken = read_dump("sio2_3_bondbreak_1.dump",
                                snapshot_indices = [0],
                                sort = True)

        box = np.copy(dump_init["box dimensions"])
        box[0,0,1] = (1+strain)*box[0,0,1]

        # merge both dump files into snapshot series
        dump = {"labels": dump_init["labels"],
                "snapshots": np.append(dump_init["snapshots"],
                                       dump_broken["snapshots"],
                                       axis=0),
                "box dimensions": np.append(dump_init["box dimensions"],
                                            dump_broken["box dimensions"],
                                            axis=0)}

        # get column ids
        x = dump["labels"].index('xu')
        z = int(dump["labels"].index('zu') + 1)
        type_id = dump["labels"].index('type')
        atom_id = dump["labels"].index('id')

        dump["atom id"] = np.append(dump_init["snapshots"][:,:,atom_id],
                             dump_broken["snapshots"][:,:,atom_id],
                             axis=0)

        # wrap coordinates along periodic boundaries
        box_lengths=dump["box dimensions"][:,:,1]-dump["box dimensions"][:,:,0]
        mask = dump["snapshots"][:,:,x:z] > dump["box dimensions"][:,None,:,1]
        dump["snapshots"][:,:,x:z] = dump["snapshots"][:,:,x:z] - mask * box_lengths[:,None,:]
        mask = dump["snapshots"][:,:,x:z] < dump["box dimensions"][:,None,:,0]
        dump["snapshots"][:,:,x:z] = dump["snapshots"][:,:,x:z] + mask * box_lengths[:,None,:]

        # calculate features for zero snapshot
        rad = symmetry_funcs_general(coordinates = dump["snapshots"][0,:,x:z],
                                     types = dump["snapshots"][0,:,type_id],
                                     box = dump["box dimensions"][0,:,:],
                                     cutoff = cutoff,
                                     rad_center = rad_center,
                                     rad_width = rad_width,
                                     pbc = True)

        # apply affine transformation to zero snapshot
        aff = box_affine_transformation(dump["snapshots"][0,:,x:z],
                                    np.append(dump_init["box dimensions"],
                                              box,
                                              axis=0))

        # calculate features for transformed snapshots
        rad_shear = symmetry_funcs_general(
                                      coordinates = aff[0],
                                      types = dump["snapshots"][0,:,type_id],
                                      box = box[0,:,:],
                                      cutoff = cutoff,
                                      rad_center = rad_center,
                                      rad_width = rad_width,
                                      pbc = True)

        # get detected first bond break
        broken = get_broken()

        # concatenate features for prediction
        features = np.concatenate((rad, rad_shear),axis=-1)

        prediction = svc.predict(features)

        # concatenate features for prediction
        features = np.concatenate((rad, rad_shear),axis=-1)

        prediction = svc.predict(features)

        # get false positives
        label = ~broken * prediction * 4

        # get true positives
        label += broken * prediction * 3

        # get false negatives
        label += broken * ~prediction * 2

        # get true negatives
        label += ~broken * ~prediction * 1

        # unite predictions and atoms
        dump_data = np.concatenate((dump["snapshots"][0,:,atom_id:atom_id+1],
                                    dump["snapshots"][0,:,x:z],
                                    np.expand_dims(label,axis=1)),axis=1)

        #
        ncol = np.shape(dump_data)[1]
        with open(os.path.join(home,os.getcwd().rsplit('/',2)[-2]+'-'+directory+'-prediction.dump'),'w') as dumpfile:

            # write timestep 0 because ovito requires it
            dumpfile.write('ITEM: TIMESTEP\n')
            dumpfile.write('0\n')

            dumpfile.write('ITEM: NUMBER OF ATOMS\n')
            dumpfile.write(str(np.shape(dump_data)[0])+'\n')

            # write box information
            dumpfile.write('ITEM: BOX BOUNDS pp pp pp\n')
            for dim in range(np.shape(dump["box dimensions"])[1]):
                dumpfile.write(str(dump["box dimensions"][0,dim,0]) +' ')
                dumpfile.write(str(dump["box dimensions"][0,dim,1]) +'\n')

            # write header for columns
            dumpfile.write('ITEM: ATOMS id x y z p \n')

            for atom in range(np.shape(dump_data)[0]):

                for col in range(ncol):
                    dumpfile.write(str(dump_data[atom,col])+' ')

                dumpfile.write('\n')

    return

if __name__ == "__main__":
    train()
