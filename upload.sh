#!/usr/bin/env sh

python setup.py bdist_wheel
twine upload dist/*
