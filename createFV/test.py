# SuperFastPython.com
# example getting the number of cpu cores
from multiprocessing import cpu_count
# get the number of logical cpu cores
n_cores = cpu_count()
# report the number of logical cpu cores
print(f'Number of Logical CPU cores: {n_cores}')