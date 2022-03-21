#!/usr/bin/env sh

ecnucs=ubuntu@192.168.10.162
deployBase=/home/ubuntu/GAIA/cs-platform
pythonExec=/home/ubuntu/anaconda3/envs/py38/bin/python

echo start deploying...
echo kill running service...
ssh ${ecnucs} "if [ -f \"$deployBase/ckqa/pid ]; then kill \$(cat $deployBase/ckqa/pid); fi"

echo clean resources...
ssh ${ecnucs} "rm -rf $deployBase/ckqa"
ssh ${ecnucs} "if [ ! -d \"$deployBase/ckqa\" ]; then mkdir -p $deployBase/ckqa; fi"
scp -r ./* ${ecnucs}:$deployBase/ckqa

echo start service...
ssh ${ecnucs} "cd $deployBase/ckqa;nohup $pythonExec server.py > nohup.out 2> nohup.err &"

echo deployment completed
