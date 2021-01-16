#!/bin/bash
DOCKER_REPO=docker-arcluster-dev.dr.corp.adobe.com
IMAGE=$USER/karima_genetic:v10

USERNAME=$(whoami)
GROUP=$(groups | awk '{print $$1}')
UID=$(id -u)
GID=$(id -g)

echo "User $USERNAME:$GROUP($UID:$GID)"

docker build -f Dockerfile.cluster \
    -t ${DOCKER_REPO}/${IMAGE} . && \
    docker push ${DOCKER_REPO}/${IMAGE}
