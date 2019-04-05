#!/usr/bin/env bash
virtualenv --python=python3 env
./env/bin/pip install -U pip
./env/bin/pip install -r requirements.txt
