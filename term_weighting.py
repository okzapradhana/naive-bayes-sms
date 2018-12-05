#Will be executed after Preprocessing
#Formula: Wt,d = log(1+ tf,d)*log10(N/dft)

import numpy as np
import preprocessing as pre

N = np.zeros(3)

N[0] = pre.getRows(pre.docTest0)
N[1] = pre.getRows(pre.docTest1)
N[2] = pre.getRows(pre.docTest2)

print(N)