import numpy as np

def read_histogram(filename, index=[0], header=False):
    """
    Reads histogram

    Parameters
    ----------
    filename: str
              full name of the histogram file
    index: list of int
           column number used as index, default 0
    header: bool
            Does the file contain a header with column names? Default False.

    Return
    ------
    hist: dict of columns
          histogram, contains 'index' column
    """

    with open(filename, 'r') as f:

        if header:
            columns = re.findall('\S+', f.readline())

        hist_arr = []
        for line in iter(f.readline, ''):
            hist_arr.append(list(map(float, re.findall('\S+', line))))

        hist_arr = np.array(hist_arr)

        if not header:
            columns = [str(i) for i in range(1, hist.shape[1])]

        hist = {}
        for i, col in enumerate(columns):
            hist[col] = hist

    return hist
