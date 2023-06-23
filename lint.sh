#!/bin/sh
result=0
flake8 emodels/ tests/ --application-import-names=emodels --import-order-style=pep8
result=$(($result | $?))
mypy --ignore-missing-imports --check-untyped-defs emodels/ tests/
result=$(($result | $?))
exit $result
