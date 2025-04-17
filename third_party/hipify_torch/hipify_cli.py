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
from hipify_torch import hipify_python

def main():
    parser = argparse.ArgumentParser(
        description='Top-level script for HIPifying, filling in most common parameters')
    parser.add_argument(
        '--project-directory',
        type=str,
        help="The root of the project. (default: %(default)s)")

    parser.add_argument(
        '--output-directory',
        type=str,
        default=None,
        help="The Directory to Store the Hipified Project",
        required=False)

    parser.add_argument(
        '--list-files-only',
        action='store_true',
        help="Only print the list of hipify files.")

    parser.add_argument(
        '--header-include-dirs',
        default=[],
        help="Directories to add to search path for header includes",
        required=False)

    parser.add_argument(
        '--includes',
        default=['*'],
        help="Source files to be included for hipify",
        required=False)

    parser.add_argument(
        '--ignores',
        default=[],
        help="Source files to be excluded for hipify",
        required=False)

    parser.add_argument(
        '--custom-map-json',
        type=str,
        help="path of json which contains project specific hipifying mappings",
        required=False)

    parser.add_argument(
        '--dump-dict-file',
        default='hipify_output_dict_dump.txt',
        type=str,
        help="The file to Store the return dict output after hipification",
        required=False)

    parser.add_argument(
        '--config-json',
        type=str,
        help="relative path of hipify config json which contains arguments to hipify",
        required=False)


    args = parser.parse_args()
    if(args.config_json):
        if(os.path.exists(args.config_json)):
            with open(args.config_json) as jsonf:
                json_args = json.load(jsonf)
                if(json_args.get('project_directory') is not None):
                    project_directory = os.path.join(os.path.dirname(args.config_json), json_args['project_directory'])
                else:
                    raise ValueError('relative path to project_dir to config_json should be mentioned')
                if(json_args.get('output_directory') is not None):
                    output_directory = os.path.join(os.path.dirname(args.config_json), json_args['output_directory'])
                else:
                    output_directory = project_directory
                if(json_args.get('includes') is not None):
                    includes = json_args['includes']
                else:
                    includes = ['*']
                if(json_args.get('header_include_dirs') is not None):
                    header_include_dirs = json_args['header_include_dirs']
                else:
                    header_include_dirs = []
                if(json_args.get('ignores') is not None):
                    ignores = json_args['ignores']
                else:
                    ignores = []
                custom_map_list=json_args.get("custom_map_json", "")
                if(json_args.get('extra_files') is not None):
                    extra_files = json_args['extra_files']
                else:
                    extra_files = []
                if(json_args.get('hipify_extra_files_only') is not None):
                    hipify_extra_files_only = json_args['hipify_extra_files_only']
                else:
                    hipify_extra_files_only = False
        else:
            raise ValueError('config json file specified should be a valid file path')
    else:
        if args.project_directory is not None:
            project_directory=args.project_directory;
        else:
            raise ValueError('If not using config json , project_directory should be mentioned in commadline')
        if args.output_directory:
            output_directory = args.output_directory
        else:
            output_directory = args.project_directory
        includes=args.includes
        ignores=args.ignores if type(args.ignores) is list \
            else args.ignores.strip("[]").split(";")
        header_include_dirs=args.header_include_dirs if type(args.header_include_dirs) is list \
            else args.header_include_dirs.strip("[]").split(";")
        custom_map_list=args.custom_map_json or ""
        extra_files = []
        hipify_extra_files_only = False
    dump_dict_file = args.dump_dict_file
    print("project_directory :",project_directory , " output_directory: ", output_directory, " includes: ", includes, " ignores: ", ignores, " header_include_dirs: ", header_include_dirs)

    HipifyFinalResult = hipify_python.hipify(
        project_directory=project_directory,
        output_directory=output_directory,
        includes=includes,
        ignores=ignores,
        header_include_dirs=header_include_dirs,
        custom_map_list=custom_map_list,
        extra_files=extra_files,
        is_pytorch_extension=True,
        hipify_extra_files_only=hipify_extra_files_only,
        show_detailed=True)

    if dump_dict_file:
        with open(dump_dict_file, 'w') as dict_file:
            dict_file.write(json.dumps(HipifyFinalResult, default=lambda o: o.asdict()))
    else:
        raise ValueError ('dump_dict_file should be defined')

if __name__ == "__main__":
    main()
