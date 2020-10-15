from mpi4py import MPI
import numpy as np
from XYMC import *

comm = MPI.COMM_WORLD 
size = comm.Get_size()
rank = comm.Get_rank()

# params to change
J = 1
T_min = 0.4
T_max = 2.2
T_per_process = 15
lattice_shape = (25, 25)
steps = 10**6
random_state = 5

Ts = np.linspace(T_min, T_max, size * T_per_process)[T_per_process * rank: T_per_process * (rank + 1)]

V = []
M2 = []
C = []

for T in Ts:
    sim = XYMetropolis(lattice_shape=lattice_shape, beta=1/T, J=J, random_state=random_state)
    sim.simulate(steps)

    print(f'Process {rank} completed {T} sim')

    V.append(sim.Vdensity)
    M2.append(sim.M2)
    C.append(sim.C)


M2 = np.array(M2)
V = np.array(V)
C = np.array(C)

print(f'Process {rank} finished all sims')

M2_tot = None
V_tot = None
C_tot = None

if rank == 0:
    M2_tot = np.empty([size, T_per_process], dtype=np.float)
    V_tot = np.empty([size, T_per_process], dtype=np.float)
    C_tot = np.empty([size, T_per_process, C.shape[1]], dtype=np.float)

comm.Gather(M2, M2_tot, root=0)
comm.Gather(V, V_tot, root=0)
comm.Gather(C, C_tot, root=0)

if rank == 0:
    Ts = np.linspace(T_min, T_max, size * T_per_process)
    VM2 = np.concatenate((Ts.reshape(size * T_per_process, 1),
                           M2_tot.flatten().reshape(size * T_per_process, 1),
                           V_tot.flatten().reshape(size * T_per_process, 1),
                           ),
                          axis=1)
    C_tot = C_tot.reshape(size*T_per_process, C.shape[1])
    np.save('VM2.npy', VM2)
    np.save('C_tot.npy', C_tot)
