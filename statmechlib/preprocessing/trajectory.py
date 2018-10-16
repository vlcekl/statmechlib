import sys
import re
import numpy as np
import h5py
import copy

class Trajectory:
    """
    Class that stores a trajectory of particle coordinates with methods for
    manipulation of the trajectory data (e.g., append, merge, sample, select, ...)

    The goal is for the class to behave similar to DataFrame of pandas.
    
    Note: will need to be refined - divide different trajectory data to
    attributes applicable to the trajectory as a whole and data. This is
    similar to hdf5 format.
    """

    def __init__(self, traj, inplace=True):
        """
        Creates a trajectory object from a dictionary or another trajectory object
        """

        if traj is None:
            self._trajectory = {}

        elif isinstance(traj, dict):
            if inplace:
                self._trajectory = traj
            else:
                self._trajectory = copy.deepcopy(traj)

        elif isinstance(traj, Trajectory):
            if inplace:
                self._trajectory = traj._trajectory
            else:
                self._trajectory = copy.deepcopy(traj._trajectory)


    def __getitem__(self, key):
        return self._trajectory[key]

    def merge(self, traj_object):
        pass

    def append(self, new_traj):
        """
        Appends a trajectory object. Tries to append only features that change
        during the simulation (currently NPT - keep particles types and numbers constant).

        Appending a large number of trajectories can be done by concat function.
        """

        all_keys = set(list(self._trajectory) + list(new_traj._trajectory))

        for key in all_keys:

            # assume appending a trajectory with the same composition
            #this is valid only for constant number of particles and types (will need to
            # be more general in future
            if ('atom' not in key) and ('ensemble' not in key) and ('0' not in key):
                self._trajectory[key] += new_traj._trajectory[key]


    def replicate(self, vec_a=1, vec_b=1, vec_c=1, inplace=False):
        """
        Replicates atomic configurations and corresponding box sizes and
        energies in given directions.

        Parameters
        ----------
        vec_{abc}: int
                   number of replications in the direction of the lattice
                   vectors a, b, and c. All default to 1.
        inplace: bool
                 If False (default), returns a new trajectory object,
                 otherwise replicated configurations stored in the current
                 object.
        
        Returns
        -------
        trj_handle: Trajectory object
                    If inplace=False, returns a new trajectory object with replicated configurations
        """

        # set trajectory object handle
        if inplace:
            trj_handle = self
        else:
            trj_handle = Trajectory(self, inplace=False)
            trj_handle._trajectory = copy.deepcopy(self._trajectory)

        # number of original atoms
        nat = sum(self['atom_num'])

        # system size scaling
        multiply = vec_a*vec_b*vec_c

        # multiply atom numbers and types
        trj_handle._trajectory['atom_type'] = self['atom_type']*multiply
        trj_handle._trajectory['atom_num'] = self['atom_num']*multiply

        # cycle over trajectory configurations
        for i in range(len(self['box'])):

            # create a new array accommodating the replicated configuration
            new_xyz = np.empty((nat*multiply, 3), dtype=float)

            # replicate coordinates, add new atoms at the end of the xyz array
            j = 0
            for ia in range(vec_a):
                for ib in range(vec_b):
                    for ic in range(vec_c):
                        new_xyz[j*nat:(j+1)*nat,0] = self['xyz'][i][0:nat,0] + float(ia)
                        new_xyz[j*nat:(j+1)*nat,1] = self['xyz'][i][0:nat,1] + float(ib)
                        new_xyz[j*nat:(j+1)*nat,2] = self['xyz'][i][0:nat,2] + float(ic)
                        j += 1
                                
            trj_handle._trajectory['xyz'][i] = new_xyz

            # scale original coordinates
            trj_handle._trajectory['xyz'][i][:,0] /= float(vec_a)
            trj_handle._trajectory['xyz'][i][:,1] /= float(vec_b)
            trj_handle._trajectory['xyz'][i][:,2] /= float(vec_c)

            trj_handle._trajectory['box'][i][0,:] = self['box'][i][0,:]*float(vec_a)
            trj_handle._trajectory['box'][i][1,:] = self['box'][i][1,:]*float(vec_b)
            trj_handle._trajectory['box'][i][2,:] = self['box'][i][2,:]*float(vec_c)

            # multiply all energy data
            for key in [k for k in self._trajectory.keys() if 'energy' in k]:
                trj_handle._trajectory[key][i] = self[key][i]*float(multiply)

        if not inplace:
            return trj_handle



    def to_xyz(self, file_name):
        """
        Save trajectory to .xyz file format
        """

        with open(file_name, 'w') as f:
            # cycle through configurations in trajectory, assign atom names, and write to file

            for box, xyz in zip(self._trajectory['box'], self._trajectory['xyz']):

                # write total number of atoms
                nat = sum(self._trajectory['atom_num'])
                f.write(f'{nat}\n')

                # write box parameters
                lx, ly, lz = box[0,0], box[1,1], box[2,2]
                xy, xz, yz = box[0,1], box[0,2], box[1,2]
                f.write(f'{lx} {ly} {lz} {xy} {xz} {yz}\n')

                # write atom coordinates
                i = 0
                for atom_type, atom_num in zip(self._trajectory['atom_type'], self._trajectory['atom_num']):
                    for _ in range(atom_num):
                        x, y, z = (box.T).dot(xyz[i])
                        f.write(f'{atom_type} {x} {y} {z}\n')
                        i += 1

