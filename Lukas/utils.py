import numpy as np
import segypy 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy

def load_seismic(filename, inlines=[1300, 1502, 2], xlines=[1500, 2002, 2]):
    inl = np.arange(*inlines)
    crl = np.arange(*xlines)
    seis, header, trace_headers = segypy.readSegy(filename)
    amplitude = seis.reshape(header['ns'], inl.size, crl.size)
    lagtime = trace_headers['LagTimeA'][0]*-1
    twt = np.arange(lagtime, header['dt']/1e3*header['ns']+lagtime, header['dt']/1e3)
    return amplitude, twt


def load_horizon(filename, inlines=[1300, 1502, 2], xlines=[1500, 2002, 2]):
    inl = np.arange(*inlines)
    crl = np.arange(*xlines)
    hrz = np.recfromtxt(filename, names=['il','xl','z'])
    horizon = np.zeros((len(inl), len(crl)))
    for i, idx in enumerate(inl):
        for j, xdx in enumerate(crl):
            time = hrz['z'][np.where((hrz['il']== idx) & (hrz['xl'] == xdx))]
            if len(time) == 1:
                horizon[i, j] = time 

    return horizon

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def interpolate_horizon(horizon):
    points = []
    wanted = []
    for i in range(horizon.shape[0]):
        for j in range(horizon.shape[1]):
            if horizon[i, j] != 0.:
                points.append([i, j, horizon[i, j]])
            else:
                wanted.append([i, j])
    
    points = np.array(points)
    zs2 = scipy.interpolate.griddata(points[:, 0:2], points[:, 2], wanted, method="cubic")
    for p, val in zip(wanted, zs2):
        horizon[p[0], p[1]] = val
    
    return horizon

def plot_section_horizon_and_well(ax, amplitude, horizon, twt, inline=38, well_pos=276//2):
    hrz_idx = [np.abs(twt-val).argmin() for val in horizon[inline, :]]
    
    h_bin = np.zeros((amplitude.shape[0], amplitude.shape[2]))
    for i, val in enumerate(hrz_idx):
        h_bin[val, i] = 1

    clip = abs(np.percentile(amplitude, 0.8))
    ax.imshow(amplitude[:, inline], cmap="Greys", vmin=-clip, vmax=clip)
    ax.plot(range(len(hrz_idx)), hrz_idx, linewidth=5, color="black")
    ax.axvline(well_pos, color="red", linewidth=5)

def flatten_on_horizon(amplitude, horizon, twt, top_add=12, below_add=52):
    traces = np.zeros((horizon.shape[0], horizon.shape[1], top_add+below_add))
    for i in range(horizon.shape[0]):
        hrz_idx = [np.abs(twt-val).argmin() for val in horizon[i, :]]
        for j in range(horizon.shape[1]):
            traces[i, j, :] = amplitude[hrz_idx[j]-top_add:hrz_idx[j]+below_add, i, j]

    return traces