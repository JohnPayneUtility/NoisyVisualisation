# NoisyVisualisation

## Dashbaord

## Generating Data

### Generating Algorithm (STN) Data

### Generating Local Optima Network (LON) Data

There are seperate run scripts for different types of LON.

LONs can be generated with a single processor using run_lon.py or with multi-processing using run_lon_parallel.py.
This requires setting parallel to true and specifying the number of processors to use in the configuration file.

'''
run:
  num_runs: 1000  # how many basin-hops (seeds) to aggregate
  seed: 1
  parallel: true
  num_workers: 50
'''

There are CoLON run scripts for Constrained Local Optima Network generation.
(The CoLON script should be able to generate regular LONs if a violation function is not provided - however this has not been tested)

python run_colon_parallel.py --config-path configs/test01_SO_KP_LON --config-name kp10_colon

