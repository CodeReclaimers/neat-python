from __future__ import print_function

import gym
import os
import sys

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: upload.py <api key> [writeup URL]")
    sys.exit(1)

ak = sys.argv[1]

# Load the results directory file, which is assumed to live in the same directory as this script.
local_dir = os.path.dirname(__file__)
results_path = os.path.abspath(os.path.join(local_dir, 'results'))

wu = None
if len(sys.argv) > 2:
    wu = sys.argv[2]
    print("Uploading results directory {0} with API key {1} and writeup {2}".format(results_path, ak, wu))
else:
    print("Uploading results directory {0} with API key {1} and no writeup".format(results_path, ak))

gym.upload(results_path, writeup=wu, api_key=ak)