import segypy
import scipy
import argparse
from scipy.interpolate import interp2d
import numpy as np

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Preprocessing')
    parser.add_argument('--min_inline', type=int, default=1, metavar='N',
                        help='Minimum Inline Value')
    parser.add_argument('--max_inline', type=int, default=1, metavar='N',
                        help='Minimum Inline Value')
    parser.add_argument('--min_xline', type=int, default=1, metavar='N',
                        help='Minimum Xline Value')
    parser.add_argument('--max_xline', type=int, default=1, metavar='N',
                        help='Minimum Xline Value')
    parser.add_argument('--step_inline', type=int, default=1, metavar='N',
                        help='Step Inline Value')
    parser.add_argument('--step_xline', type=int, default=1, metavar='N',
                        help='Step Xline Value')
    parser.add_argument('--fname_near_stack', type=str, default=None, metavar='N',
                        help='File Name Near Stack')
    parser.add_argument('--fname_far_stack', type=str, default=None, metavar='N',
                        help='File Name Far Stack')
    parser.add_argument('--fname_horizon', type=str, default=None, metavar='N',
                        help='File Name Horizon')
    parser.add_argument('--out_dir', type=str, default=None, metavar='N',
                        help='output directory')
    args = parser.parse_args(argv)
    return args

def load_seismic(fname, min_inline, max_inline, step_inline, min_xline, max_xline, step_xline, scaling=1e3):
    inl = np.arange(min_inline, max_inline, step_inline)
    crl = np.arange(min_xline, max_xline, step_xline)

    seismic, header, trace_headers = segypy.readSegy(fname)
    seismic = seismic.reshape(header['ns'], inl.size, crl.size)

    lagtime = trace_headers['LagTimeA'][0]*-1
    twt = np.arange(lagtime, header['dt']/scaling*header['ns']+lagtime, header['dt']/scaling)
    return seismic, twt


def load_horizon(fname, min_inline, max_inline, step_inline, min_xline, max_xline, step_xline, method="cubic"):
    inl = np.arange(min_inline, max_inline, step_inline)
    crl = np.arange(min_xline, max_xline, step_xline)

    hrz = np.recfromtxt(fname, names=['il','xl','z'])

    horizon = np.zeros((len(inl), len(crl)))

    for i, idx in enumerate(inl):
        for j, xdx in enumerate(crl):
            time = hrz['z'][np.where((hrz['il']== idx) & (hrz['xl'] == xdx))]
            if len(time) == 1:
                horizon[i, j] = time 
    
    interpreted_horizon = horizon.copy()

    points = []
    wanted = []
    for i in range(horizon.shape[0]):
        for j in range(horizon.shape[1]):
            if horizon[i, j] != 0.:
                points.append([i, j, horizon[i, j]]) #The points where we have data
            else:
                wanted.append([i, j]) #These are the missing points
    points = np.array(points)

    #here we provide first i, j corrdinates of the points to interpolate, then the actual twt of the horizon.
    interpolated_twt = scipy.interpolate.griddata(points[:, 0:2], points[:, 2], wanted, method=method)

    for p, val in zip(wanted, interpolated_twt):
        horizon[p[0], p[1]] = val

    return interpreted_horizon, horizon

def find_closest_samples_to_horizon(horizon, twt, N):
    hrz_idx = [np.abs(twt-val).argmin() for val in horizon[N, :]]
    return hrz_idx

def extract_trace_above_below_horizon(seismic, horizon_ids, samples_above, samples_below, N, M):
    trace = seismic[horizon_ids[M]-samples_above:horizon_ids[M]+samples_below, N, M]
    return trace

def extract_chunk(seismic, twt, horizon, samples_above, samples_below):
    traces = np.zeros((seismic.shape[1], seismic.shape[2], samples_above+samples_below))
    for i in range(seismic.shape[1]):
        hrz_idx = [np.abs(twt-val).argmin() for val in horizon[i, :]]
        for j in range(seismic.shape[2]):
            amplitude = seismic[hrz_idx[j]-samples_above:hrz_idx[j]+samples_below, i, j]
            traces[i, j, :] = amplitude
    return traces

def main(args):
    pass

if __name__=="__main__":
    args = parse_args(sys.argv[1:])
    main(args)




