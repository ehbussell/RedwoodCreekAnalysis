""" Main options common to many Redwood Creek analysis scripts. """

import numpy as np

OPTIONS = {
    'eroi': ((-124.15045, 40.9542), (-123.7493, 41.2364)),
    'roi': ((-124.1142, 40.9783), (-123.7857, 41.2117)),
    'times': np.linspace(0, 30, 421),
    'control_rate': 8
}
