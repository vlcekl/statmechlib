import numpy as np

def read_histogram(filename, header=False):
    """
    Reads histogram

    Parameters
    ----------
    filename: str
              full name of the histogram file
    header: bool
            Does the file contain a header with column names? Default False.

    Return
    ------
    hist: ndarray of float
          histograms
    columns: list of ints or strings
          column identifiers
    """

    with open(filename, 'r') as f:
        if header:
            columns = re.findall('\S+', f.readline())

        hist = []
        for line in iter(f.readline, ''):
            hist.append(list(map(float, re.findall('\S+', line))))

        hist = np.array(hist)

        if not header:
            columns = [i for i in range(hist.shape[1])]

    return np.array(hist), columns
