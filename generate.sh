#!/usr/bin/env sh

rm ./service/gen-py/* -f
thrift -o ./service -r --gen py ckqa.thrift
