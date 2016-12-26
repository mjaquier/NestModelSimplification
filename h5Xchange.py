import numpy as np
import glob
import h5py
import time


RECORDING_DIR = '### FILL IN ###'
PATH_FLAT = '### FILL IN ###'
EXPERIMENT_PATH = '### FILL IN ###'
DATA_PATH = glob.glob(PATH_FLAT+EXPERIMENT_PATH+"*.dat")
H5_PATH = PATH_FLAT+EXPERIMENT_PATH+'### FILL IN ###'

V = []
I = []
T = []
for i, k in enumerate(DATA_PATH):

    tempT = np.loadtxt(k, usecols=[0])
    tempV = np.loadtxt(k, usecols=[1])
    tempA = np.loadtxt(k, usecols=[2])
    T.append(tempT)
    V.append(tempV)
    I.append(tempA)

timestr = time.strftime("%H%M%S")
print(np.size(T) > 9e4)
if np.size(T) > 9e5:
    fpath = H5_PATH+'Train/'+'dataFile{}.hdf5'.format(timestr)
else:
    fpath = H5_PATH+'Test/'+'dataFile_{}.hdf5'.format(timestr)

data = h5py.File('{}'.format(fpath), 'w')
volt_dset = data.create_dataset('voltage', data=V, compression='gzip')
amp_dset = data.create_dataset('current', data=I, compression='gzip')
time_dset = data.create_dataset('time', data=T, compression='gzip')
data.close()


# Safety

mega = h5py.File('{}'.format(fpath), 'r')
for n, k in enumerate(zip(V, mega['voltage'])):
    assert(all(V[n] == mega['voltage'][n]))
for n, k in enumerate(zip(I, mega['current'])):
    assert(all(I[n] == mega['current'][n]))
for n, k in enumerate(zip(T, mega['time'])):
    assert(all(T[n] == mega['time'][n]))
mega.close()

