#!/usr/bin/env bash

if [ -d "./service/gen-py" ]
then
  rm -rf ./service/gen-py/*
fi
thrift -o ./service -r --gen py ckqa.thrift
