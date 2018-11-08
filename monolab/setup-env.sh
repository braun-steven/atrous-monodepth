#!/usr/bin/env bash
virtualenv --python=python3 env
./env/bin/pip install -U pi
./env/bin/pip install -r requirements.txt