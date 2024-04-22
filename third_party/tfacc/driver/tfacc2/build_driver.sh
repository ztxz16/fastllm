#!/bin/bash

base_dir=$(
  cd "$(dirname "$0")" || exit
  pwd
)
cd "${base_dir}" || exit

cp ../../tfsmi /usr/local/bin/tfsmi
cp ../../tfsmbios /usr/local/bin/tfsmbios

output_path=result/$1

rm -rf "${output_path}"
mkdir -p "${output_path}"
cp -r $(ls | grep -v result | xargs) "${output_path}"

cd "${output_path}" || exit

export LINUX_VERSION=$(cat /etc/issue | awk -F ' ' '{print $1}' | awk 'NR==1')
make tfacc2
