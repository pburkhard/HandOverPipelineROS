#!/bin/bash

PROFILE="${1:-full}"

PROJECT_ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."
docker compose -f "${PROJECT_ROOT}/docker/docker-compose.yaml" --profile "$PROFILE" up -d --build