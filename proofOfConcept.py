import numpy as np
from neuron import h, rxd
from neuron.units import ms, mV
import matplotlib.pyplot as plt
import pprint
import network
import networkx as nx

rxd.options.enable.extracellular = True
rxd.nthread(4)

h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")


class Cell:
    # TODO: add tau concentration instance var
    def __init__(self, filename, x, y, z, theta):
        if filename=="b&s":
            self._setup_morphology()
            self._setup_biophysics()
        else:
            cell = h.Import3d_SWC_read()
            cell.input(filename)
            h.Import3d_GUI(cell, 0)
            i3d = h.Import3d_GUI(cell, 0)
            i3d.instantiate(self)
        self.x = self.y = self.z = 0
        self.syn = h.ExpSyn(self.dend(0.5))
        self.syn.tau = 2 * ms
        h.define_shape()
        self._rotate_z(theta)
        self._set_position(x, y, z)
        for sec in self.all:
            sec.nseg = 1 + 10 * int(sec.L / 5)

        for sec in self.all:
            sec.Ra = 100  # Axial resistance in Ohm * cm
            sec.cm = 1  # Membrane capacitance in micro Farads / cm^2
        self.soma.insert('hh')
        for seg in self.soma:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -54.3  # Reversal potential in mV
        # Insert passive current in the dendrite
        #for dend in self.dend:
        self.dend.insert('pas')
        #for dend in self.dend:
        for seg in self.dend:
            seg.pas.g = 0.001  # Passive conductance in S/cm2
            seg.pas.e = -65  # Leak reversal potential mV

    def _setup_morphology(self):
        """initialzes cell morphology"""
        self.soma = h.Section(name='soma', cell=self) # create soma and dendrite
        self.dend = h.Section(name='dend', cell=self)
        self.all = [self.soma, self.dend]
        self.dend.connect(self.soma)  # connect the two, Note that this is not equivalent to attaching the soma to the dend; instead it means that the dendrite begins where the soma ends.
        #We could explicitly specify the connection location via, e.g. self.dend.connect(self.soma(0.5)) which would mean the dendrite was attached to the center of the soma.
        self.soma.L = self.soma.diam = 12.6157  # microns
        self.dend.L = 200
        self.dend.diam = 1
    def _setup_biophysics(self):
        """initlizes cell biophysics"""
        for sec in self.all:
            sec.Ra = 100    # Axial resistance in Ohm * cm
            sec.cm = 1      # Membrane capacitance in micro Farads / cm^2
        self.soma.insert('hh')
        for seg in self.soma: # here we loop over all segments in the soma, even though we only defined one segment.
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003    # Leak conductance in S/cm2
            seg.hh.el = -54.3     # Reversal potential in mV
        # Insert passive current in the dendrite
        self.dend.insert('pas')
        for seg in self.dend:
            seg.pas.g = 0.001  # Passive conductance in S/cm2
            seg.pas.e = -65    # Leak reversal potential mV
    def _set_position(self, x, y, z):
        """sets the position"""
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(i,
                               x - self.x + sec.x3d(i),
                               y - self.y + sec.y3d(i),
                               z - self.z + sec.z3d(i),
                               sec.diam3d(i))
        self.x, self.y, self.z = x, y, z

    def _rotate_z(self, theta):
        """Rotate the cell about the Z axis."""
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                c = h.cos(theta)
                s = h.sin(theta)
                xprime = x * c - y * s
                yprime = x * s + y * c
                sec.pt3dchange(i, xprime, yprime, sec.z3d(i), sec.diam3d(i))

    def getInfo(self):
        return f"cell at location ({self.x}, {self.y})"


def createCells(matrix, r):
    n = len(matrix)
    cells = []
    for x in range(n):
        theta = x * 2 * h.PI / n
        cells.append(Cell("b&s", h.cos(theta) * r, h.sin(theta) * r, 0, theta))
    return cells


def connectCells(matrix, cells, verbose=False):
    """:returns adj matrix of netcons, [from][to]"""
    n = len(matrix)
    ncs = np.zeros((n, n), dtype=object)
    for i, cell in enumerate(cells):
        connectTo = [tgti for tgti, x in enumerate(matrix[i]) if x == 1]
        for index in connectTo:
            tgt = cells[index]
            nc = h.NetCon(cell.soma(0.8)._ref_v, tgt.syn, sec=cell.soma)
            #print(f"{nc} has originates cell {nc.precell()} and targets cell {nc.postcell()}")
            nc.weight[0] = .05
            nc.delay = 5
            ncs[i][index] = nc
            if verbose:
                print(f"Connected cell {i} to cell {index}.")
    return ncs

def countSpikes(vector):
    v = list(vector)
    c=0
    for val in v:
        if val>10:
            c+=1
    return c



def showStruc():
    ps = h.PlotShape(True)
    ps.show(0)
    input()


#net = network.small_world(50, 2, 0)
net = network.scale_free(25, 2)
network.visualize_network(net)
plt.show()
matrix = nx.to_numpy_array(net)
print(np.sum(matrix))
cells = createCells(matrix, r=600)
netcons = connectCells(matrix, cells)
#showStruc()
n = len(matrix)

stim = h.NetStim()
syn = h.ExpSyn(cells[12].dend(0.8))
stim.number = 5
stim.start = 9

ncstim = h.NetCon(stim, syn)
ncstim.delay = 1 * ms
ncstim.weight[0] = .08

syn.tau = 2 * ms


v0 = h.Vector().record(cells[0].soma(0.5)._ref_v)
v1 = h.Vector().record(cells[1].soma(0.5)._ref_v)
v3 = h.Vector().record(cells[3].soma(0.5)._ref_v)
v5 = h.Vector().record(cells[5].soma(0.5)._ref_v)
t = h.Vector().record(h._ref_t)

spikeTimeVecs = np.zeros((n, n), dtype=object)
for fro in range(n):
    for to in range(n):
        spikeTimeVecs[fro, to] = h.Vector()
        if netcons[fro, to] != 0:
            netcons[fro, to].record(spikeTimeVecs[fro, to])

# for incNeuron in range(n):
#     ncs = netcons[incNeuron]
#     spike_times = [h.Vector() for nc in ncs]
#     for nc, spike_times_vec in zip(ncs, spike_times):
#         nc.record(spike_times_vec)
#     spk_times.append(spike_times)
#     spktimes[]
pprint.pprint(spikeTimeVecs)

pprint.pprint(netcons)
# print(spk_times)

h.finitialize(-65 * mV)
h.continuerun(300 * ms)
incspk_counts = [0] * n

# for incNeuron in range(n):
#     spike_times = spk_times[incNeuron]
#     print(incNeuron)
#     for i, spike_times_vec in enumerate(spike_times):
#         incspk_counts[incNeuron] += len(spike_times_vec)
#         print(list(spike_times_vec))

for fro in range(n):
    for to in range(n):
        pprint.pprint(f"from {fro} to {to} : {list(spikeTimeVecs[fro, to])}")

fig, ax = plt.subplots()

for fro in range(n):
    for i, spike_times_vec in enumerate(spikeTimeVecs[fro]):
        #plt.vlines(list(spike_times_vec), fro-.5, fro + .5)
        X = list(spike_times_vec)
        Y = fro*np.ones_like(X)
        ax.scatter(X, Y, color='k')
        #print(list(spike_times_vec), "from", fro, "to",i)
plt.show(block=False)

plt.figure()
plt.plot(t, v0, label='cell0')
plt.legend()
plt.show(block=False)
plt.figure()
plt.plot(t, v1, label='cell1')
plt.legend()
plt.show(block=False)
plt.figure()
plt.plot(t, v3, label='cell3')
plt.legend()
plt.show(block=False)
plt.figure()
plt.plot(t, v5, label='cell5')
plt.legend()
plt.show()
