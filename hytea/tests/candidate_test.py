
# creating combinations of 0 and 1
from itertools import product

from hytea.candidate import Candidate

import numpy as np

C1 = Candidate(np.zeros(18))

def test_nHidden():
    # loop over all combinations of 0 and 1

    label_to_bin = {
        1: (0, 0),
        2: (0, 1),
        3: (1, 0),
        4: (1, 1),
    }

    for label, bin_rep in label_to_bin.items():
        # convert tuple to numpy array
        sec = np.array(bin_rep)
        # set the first two elements of the genome to sec
        C1.genome[0:2] = sec
        assert C1.nHidden == label, \
            f"nHidden should be {label} for {sec}, but got {C1.nHidden}"
         

def test_dropOutRate():
    print(C1.dropOutRate)


if __name__ == '__main__':
    test_nHidden()
