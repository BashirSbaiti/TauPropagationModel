import random

import numpy as np
from neuron import h
from neuron.units import ms, mV
import matplotlib.pyplot as plt
import pprint
import os
from scipy.special import expit
import network
import random as rand
import networkx as nx

h.load_file("stdrun.hoc")
h.load_file("import3d.hoc")


class Cell:
    def __init__(self, filename, x, y, z, theta):
        if filename == "b&s":
            self.BnS_morphology()
            self.BnD_ephys()
        else:
            cell = h.Import3d_SWC_read()
            cell.input(filename)
            h.Import3d_GUI(cell, 0)
            i3d = h.Import3d_GUI(cell, 0)
            i3d.instantiate(self)
            for sec in self.all:
                sec.Ra = 100  # Axial resistance in Ohm * cm
                sec.cm = 1  # Membrane capacitance in micro Farads / cm^2
            self.soma[0].insert('hh')
            for seg in self.soma[0]:
                seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
                seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
                seg.hh.gl = 0.0003  # Leak conductance in S/cm2
                seg.hh.el = -54.3  # Reversal potential in mV
            for dend in self.dend:
                dend.insert('pas')
            for dend in self.dend:
                for seg in dend:
                    seg.pas.g = 0.001  # Passive conductance in S/cm2
                    seg.pas.e = -65  # Leak reversal potential mV
        self.x = self.y = self.z = 0
        self.tauC = 0.0
        self.isAlive = True
        self.syn = h.ExpSyn(self.dend(0.5)) if type(self.dend) is not list else h.ExpSyn(self.dend[0](0.5))
        self.syn.tau = 2 * ms
        h.define_shape()
        self.rotateZ(theta)
        self.set_pos(x, y, z)
        for sec in self.all:
            sec.nseg = 1 + 10 * int(sec.L / 5)

    def BnS_morphology(self):
        """initialzes cell morphology"""
        self.soma = h.Section(name='soma', cell=self)
        self.dend = h.Section(name='dend', cell=self)
        self.all = [self.soma, self.dend]
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157  # microns
        self.dend.L = 200
        self.dend.diam = 1

    def BnD_ephys(self):
        """initlizes cell biophysics"""  # TODO: compte IF curve (current v firing rate) / improve ephys?
        for sec in self.all:
            sec.Ra = 100  # Axial resistance in Ohm * cm
            sec.cm = 1  # Membrane capacitance in micro Farads / cm^2
        self.soma.insert('hh')
        for seg in self.soma:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -54.3  # Reversal potential in mV
        # passive current
        self.dend.insert('pas')
        for seg in self.dend:
            seg.pas.g = 0.001  # Passive conductance in S/cm2
            seg.pas.e = -65  # Leak reversal potential mV

    def set_pos(self, x, y, z):
        """sets the position"""
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(i,
                               x - self.x + sec.x3d(i),
                               y - self.y + sec.y3d(i),
                               z - self.z + sec.z3d(i),
                               sec.diam3d(i))
        self.x, self.y, self.z = x, y, z

    def rotateZ(self, theta):
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

    def kill(self, lists=False):
        for sec in self.all:
            sec.Ra = float('inf')
        for seg in self.soma[0] if lists else self.soma:
            seg.hh.gnabar = 0
            seg.hh.gkbar = 0
            seg.hh.gl = 0
        for dend in self.dend if lists else range(1):
            for seg in dend if lists else self.dend:
                seg.pas.g = 0
        self.isAlive = False


def createCells(matrix, r):
    n = len(matrix)
    cells = []
    for x in range(n):
        theta = x * 2 * h.PI / n
        cells.append(Cell("b&s", h.cos(theta) * r, h.sin(theta) * r, 0, theta))
    return cells


def connectCells(matrix, cells, inhFrac=0.0, verbose=False):
    """:returns adj matrix of netcons, [from][to]"""
    n = len(matrix)
    ncs = np.zeros((n, n), dtype=object)
    numInhCells = int(inhFrac * n)
    inhCellIndxs = random.sample(range(0, n), numInhCells)
    for i, cell in enumerate(cells):
        connectTo = [tgti for tgti, x in enumerate(matrix[i]) if x == 1]
        for index in connectTo:
            tgt = cells[index]
            nc = h.NetCon(cell.soma(0.8)._ref_v, tgt.syn, sec=cell.soma) if type(cell.soma) is not list else h.NetCon(
                cell.soma[0](0.8)._ref_v, tgt.syn, sec=cell.soma[0])
            # print(f"{nc} has originates cell {nc.precell()} and targets cell {nc.postcell()}")
            nc.weight[0] = .05 if i not in inhCellIndxs else -0.05
            nc.delay = 5
            ncs[i][index] = nc
            if verbose:
                print(f"Connected cell {i} to cell {index} with a connection weight of {nc.weight[0]}")
    return ncs


onTheBooks = True

# network parameters
nn = 1000
kn = 20
pn = .05
inhFrac = 0.15

# time control
timeSteps = 20
time = 500 * ms
rmp = -65
dt = time / timeSteps

# therapeutic parameters
reducFactors = [.2, .5, .7, .9]
props = [.2, .5, .7]
treatDelay = 0

# tau parameters
lethalThreshold = .65
initialFrac = .2
initialConc = .1

# initial simulation parameters
number = 5
start = 9
stimDelay = 1
weight = 0.08
tau = 2

total = 12
progress = 0

for reducFactor in reducFactors:
    for prop in props:

        def showStruc():
            ps = h.PlotShape(True)
            ps.show(0)
            input()


        def getSpikeCounts(matrix, spikeTimeVecs, tstart, tstop, verbose=False):
            """returns a matrix shape=(i,j) of spike counts for cell i firing to cell j in the interval (tstart, tstop)
            will also copy firing times of cell a to cell b to all of the cells cell a fires too, because neuron
            will only keep track of one"""
            n = len(matrix)
            firingMat = np.zeros_like(matrix)
            for fro in range(n):
                for to in range(n):
                    spikeList = list(spikeTimeVecs[fro][to])
                    if h.t >= 194 and verbose:
                        print(spikeList, h.t, tstart, tstop, f"cell {fro} to {to}")
                    spikeListInInterval = [el for el in spikeList if tstart < el < tstop]
                    if len(spikeListInInterval) > 0:
                        firingMat[fro] = np.array(matrix[fro]) * len(spikeListInInterval)
            return firingMat


        def seedTau(cells, initial_frac=0.1, intial_conc=0.1):
            n = len(cells)
            toInfect = int(initial_frac * n)
            infectedCount = 0

            while infectedCount < toInfect:
                rIndex = int(rand.random() * n)
                if cells[rIndex].tauC == 0:
                    cells[rIndex].tauC = intial_conc
                    infectedCount += 1


        def propogateTau(cells, FRMatrix, verbose=False):
            for i, cell in enumerate(cells):
                if cell.isAlive:
                    incidentCells = np.array([ind for ind, x in enumerate(FRMatrix[:, i]) if x > 0])
                    incidentFRs = np.array([x for x in FRMatrix[:, i] if x > 0])
                    incidentPercentFiringAc = 0 if np.sum(incidentFRs) == 0 else incidentFRs / np.sum(incidentFRs)
                    incidentTaus = np.zeros(len(incidentCells))
                    for ind, i2 in enumerate(incidentCells):
                        incidentTaus[ind] = cells[i2].tauC
                    dTau = np.sum(incidentPercentFiringAc * incidentTaus)
                    if verbose:
                        print(
                            f"{incidentTaus}\t{incidentFRs}\t{dTau} for cell {i} which currently has tau = {cell.tauC} Time = {h.t}")
                    cell.tauC = sigmoid(cell.tauC + dTau) if dTau > 0 else cell.tauC


        def sigmoid(x):
            return expit(x)


        matrix = network.small_world(nn, kn, pn)
        # network.visualize_matrix(matrix)
        # plt.show()
        print(f"Network has {np.sum(matrix)} total connections")
        cells = createCells(matrix, r=600)
        netcons = connectCells(matrix, cells, inhFrac=inhFrac, verbose=False)
        # showStruc()
        n = len(matrix)


        def killCellsAbove(threshold, verbose=False):
            """kills cells with a Tau concentration above threshold"""
            killCount = 0
            for i, cell in enumerate(cells):
                # print(f"cell {i} has tauC = {cell.tauC}, time = {h.t}")
                if cell.tauC > threshold and cell.isAlive:
                    cell.kill()
                    killCount += 1
                    if verbose:
                        print(f"killed cell {i} at {h.t}")
            return killCount


        def deliverTherapeutic(TauReducFactor, prop, delay, verbose=False):
            """reduced tau concentration by TauReducFactor (percent) in prop percent of infected cells"""
            if h.t >= delay:
                threshold = 0.001  # concentration that is close enough to zero
                infectedCellsInds = [i for i, cell in enumerate(cells) if cell.tauC > threshold and cell.isAlive]
                random.shuffle(infectedCellsInds)
                cellsToAffect = infectedCellsInds[0:int(prop * len(infectedCellsInds))]
                # print(cellsToAffect, len(cellsToAffect))

                for ind in cellsToAffect:
                    # if it was > 0, would go on forever
                    if cells[ind].tauC > threshold and cells[ind].isAlive:  # TODO affect dead cells?
                        if verbose:
                            print(f"cell {ind}'s tau concentration reduced from {cells[ind].tauC} to ", end="")
                        cells[ind].tauC -= TauReducFactor * cells[ind].tauC
                        if verbose:
                            print(f"{cells[ind].tauC} at {h.t}")


        def initialStim(number, start, delay, weight, tau):
            stim = h.NetStim()
            syn = h.ExpSyn(cells[n // 2].dend(0.8)) if type(cells[n // 2].dend) is not list else h.ExpSyn(
                cells[n // 2].dend[0](0.8))
            stim.number = number
            stim.start = start

            ncstim = h.NetCon(stim, syn)
            ncstim.delay = delay * ms
            ncstim.weight[0] = weight

            syn.tau = tau * ms

            return ncstim, stim, syn


        def runSim(Vm, tstop, timesteps):
            """runs the simulation"""
            h.finitialize(Vm * mV)
            dt = tstop / timesteps
            events = list()
            for ts in range(timesteps):
                events.append(h.CVode().event(ts * dt, step))
            h.continuerun(tstop * ms)


        ncstim, stim, syn = initialStim(number, start, stimDelay, weight, tau)


        def recordV(cellNumber, region="soma"):
            """records the voltage of cellNumber"""
            if region == "soma":
                return h.Vector().record(
                    cells[cellNumber].soma(0.5)._ref_v if type(cells[cellNumber].soma) is not list
                    else cells[cellNumber].soma[0](0.5)._ref_v)
            else:
                return h.Vector().record(
                    cells[cellNumber].dend(0.5)._ref_v if type(cells[cellNumber].dend) is not list
                    else cells[cellNumber].dend[0](0.5)._ref_v)


        v = recordV(n//2, region="soma")
        t = h.Vector().record(h._ref_t)

        spikeTimeMat_Unexpanded = np.zeros((n, n), dtype=object)
        for fro in range(n):
            for to in range(n):
                spikeTimeMat_Unexpanded[fro, to] = h.Vector()
                if netcons[fro, to] != 0:
                    netcons[fro, to].record(spikeTimeMat_Unexpanded[fro, to])

        seedTau(cells, initial_frac=initialFrac, intial_conc=initialConc)
        tauMatrix = []
        numDeadCells = []


        def step(dt=5, verbose=False):
            """performs one timestep"""
            spikeTimes = getSpikeCounts(matrix, spikeTimeMat_Unexpanded, h.t - dt, h.t, verbose=False)
            if verbose:
                print(f"spikes in interval = {np.sum(spikeTimes)}")

            propogateTau(cells, spikeTimes, verbose=False)  # PAY ATTENTION TO ORDER
            deliverTherapeutic(reducFactor, prop, treatDelay, verbose=False)

            killed = killCellsAbove(lethalThreshold, verbose=False)
            global numDeadCells
            if len(numDeadCells) == 0:
                numDeadCells.append(killed)
            else:
                numDeadCells.append(killed + numDeadCells[-1])

            global tauMatrix
            taus = np.array([cell.tauC for cell in cells]).reshape(n, 1)
            if len(tauMatrix) == 0:
                tauMatrix = taus
            else:
                tauMatrix = np.hstack((tauMatrix, taus))


        runSim(rmp, time, timeSteps)


        def plotTauPrev(folder, plot=True, save=False):
            """plots the tau prevalence over time (total tau concentration/number of cells, for each timestamp)"""
            plt.figure()
            plt.title("Average Cellular Tau over Time")
            plt.xlabel('Time (ms)')
            plt.ylabel('Average Cellular Tau (0-1)')
            tauPrev = np.sum(tauMatrix, axis=0) / n
            plt.plot(np.arange(0, timeSteps, 1) * dt, tauPrev)
            if (save):
                plt.savefig(f"Results/{folder}/tauPrevalence.png")
            if (plot):
                plt.show()


        def plotSpikeTimes(plotType, folder, plot=True, save=False):
            """plots the spike times of all cells in the simulation
            :param plotType (string): raster or scatter"""
            fig, ax = plt.subplots()
            plt.title("Spike Times of Cells in Network")
            plt.xlabel('Time (ms)')
            plt.ylabel('Cell Number')

            for fro in range(n):
                for i, spike_times_vec in enumerate(spikeTimeMat_Unexpanded[fro]):
                    X = list(spike_times_vec)
                    if len(X) == 0:
                        continue
                    if plotType == "raster":
                        plt.vlines(X, fro - .5, fro + .5)
                    elif plotType == "scatter":
                        Y = fro * np.ones_like(X)
                        ax.scatter(X, Y, color='k')
                    else:
                        return
            if save:
                plt.savefig(f"Results/{folder}/spikeTimePlot_{plotType}.png")
            if not plot:
                return
            plt.show()


        def plotMemPotential(cellNumber, folder, plot=True, save=False):
            """plots the membrane potential of cellNumber over time"""
            plt.figure()
            plt.title(f"Membrane Potential of Cell {cellNumber} Over Time")
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Potential (mV)')
            plt.plot(t, v)
            if (save):
                plt.savefig(f"Results/{folder}/MembranePotentialCell{cellNumber}.png")
            if (plot):
                plt.show()


        def plotCellSurvival(survivalList, folder, plot=True, save=False):
            """plots number of cells alive at each timestep
            :param survivalList: list of cells alive at each timestep"""
            plt.figure()
            plt.title(f"Number of Surviving Cells over Time")
            plt.xlabel('Time (ms)')
            plt.ylabel('Surviving Cells')
            plt.plot(np.arange(0, timeSteps, 1) * dt, survivalList)
            if (save):
                plt.savefig(f"Results/{folder}/cellSurvival.png")
            if (plot):
                plt.show()


        if onTheBooks:
            foldername = f"n{nn}k{kn}p{pn}rf{reducFactor}tp{prop}td{treatDelay}le{lethalThreshold}if{inhFrac}t{time}ts{timeSteps}"
            os.makedirs(f"Results/{foldername}/")
            plotTauPrev(foldername, plot=False, save=True)
            plotMemPotential(n//2, foldername, plot=False, save=True)

            survivalList = n - np.array(numDeadCells)
            plotCellSurvival(survivalList, foldername, plot=False, save=True)

            with open("Results/params.txt", "a") as f:
                f.write(f"n = {nn}\tk = {kn}\tp = {pn}\tinh frac = {inhFrac}\t")
                f.write(
                    f"STIM num = {number}\tstart = {start}\tdelay = {stimDelay}\tweight = {weight}\ttau(time const) = {tau}\t")
                f.write(f"SEED fraction = {initialFrac}\tinitial conc = {initialConc}\t")
                f.write(
                    f"STEP: reduc factor = {reducFactor}\ttreated prop = {prop}\ttreatment delay = {treatDelay}\tlethal = {lethalThreshold}\t")
                f.write(f"TIME = {time}\ttime steps = {timeSteps}\trmp = {rmp}\tdt = {dt}\n")
        else:
            plotTauPrev(None, plot=True, save=False)
            plotMemPotential(n//2, None, plot=True, save=False)
            plotSpikeTimes("raster", None, plot=True, save=False)

            survivalList = n - np.array(numDeadCells)
            plotCellSurvival(survivalList, None, plot=True, save=False)

        progress += 1
        print(f"{progress / total * 100}% done")
