#!/usr/bin/env bash

docker run \
    --privileged \
    -p 8080:8080 \
    -p 8081:8081 \
    --volume ~/scalebox:/scalebox \
    -w /scalebox \
    --health-cmd='python /scalebox/deploy/a_plus_b.py || exit 1' \
    --health-interval=2s \
    -itd \
    --restart unless-stopped \
    quay.io/jszheng/scalebox:x86-20260331 \
    make run-online