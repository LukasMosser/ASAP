import segypy
import numpy as np
# Set verbose level
segypy.verbose=1;

def load_segy(filename, min_inline, max_inline, step_inline, min_xline, max_xline, step_xline):
    # Get only SegyHeader
    SH = segypy.getSegyHeader(filename);

    inl=np.arange(min_inline, max_inline, step_inline)
    crl=np.arange(min_xline, max_xline, step_xline)

    seis, header, trace_headers = segypy.readSegy(filename)

    stack = seis.reshape(header['ns'], inl.size, crl.size)
    return stack