from typing import Union
import numpy

def normalize_array(array:Union[list, tuple, numpy.ndarray], min_val:float, max_val:float) -> numpy.ndarray:
    return numpy.array([(x - min_val) / (max_val - min_val) for x in array])