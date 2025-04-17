#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--io-file',
        type=str,
        help="Input file containing list of files which will be overwritten by hipified file names",
        required=True)

    parser.add_argument(
        '--dump-dict-file',
        type=str,
        help="Input file where the dictionary output of hipify is stored",
        required=True)

    args = parser.parse_args()

    file_obj = open(args.dump_dict_file, mode='r')
    json_string = file_obj.read()
    file_obj.close()
    hipified_result = json.loads(json_string)

    out_list = []
    with open(args.io_file) as inp_file:
        for line in inp_file:
            line = line.strip()
            line = os.path.abspath(line)
            if line in hipified_result:
                out_list.append(hipified_result[line]['hipified_path'])
            else:
                out_list.append(line)

    w_file_obj = open(args.io_file, mode='w')
    for f in out_list:
        w_file_obj.write(f+"\n")
    w_file_obj.close()

if __name__ == "__main__":
    main()
