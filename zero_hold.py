import numpy

def fractional_repeat(x, r, out_len=None):
    '''
    Upsample x by repeating its values (zero-hold)

    Parameters
    ----------
    x: array_like
        the signal to upsample
    r: float
        the ratio of the sampling rates
    out_len: int, optional
        desired output length
    '''

    if out_len is None:
        out_len = int(r * x.shape[0])

    if x.ndim == 1:
        y = numpy.empty(out_len, dtype=x.dtype)
    else:
        y = numpy.empty((out_len,) + x.shape[1:], dtype=x.dtype)

    length = 0
    i = 0
    sample_error = 0
    while length < out_len and i < x.shape[0]:

        end = min(out_len, length + int(r))
        L = end - length

        y[length:end,] = x[i,]

        length = end

        # finish early if necessary
        if end == out_len:
            break

        # keep track of fractional sample error accumulation
        sample_error += r - L

        # adjust when necessary
        if sample_error > 1.:
            y[length,] = x[i,]
            sample_error -= 1
            length += 1

        i += 1

    return y

