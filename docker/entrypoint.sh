#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.
set -euo pipefail
conda activate yolactpp-env
exec python /yolactpp/external/DCNv2/setup.py build develop
exec rm -Rf /root/.cache/pip 
