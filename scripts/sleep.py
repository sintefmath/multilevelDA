import argparse
parser = argparse.ArgumentParser(description='Going to bed')
parser.add_argument('--T', type=int, default=1)

pargs = parser.parse_args()

T = pargs.T

import time
print("Going to sleep for ", T)
time.sleep(T)
