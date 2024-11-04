#!/bin/bash
# Based on Gist: AndrejOrsula/_docker_helper_scripts.md

set -e

## Configuration
# Directory and image
IMAGE_NAME="kuldeepbrd/grasp_ldm:latest"

# Flags for running the container
DOCKER_RUN_OPTS="${DOCKER_RUN_OPTS:-
    --interactive
    --tty
    --rm
    --network host
    --ipc host
}"

# Flags for enabling GPU and GUI (X11) inside the container
ENABLE_GPU="${ENABLE_GPU:-true}"
ENABLE_GUI="${ENABLE_GUI:-true}"


# List of volumes to mount (can be updated by passing -v HOST_DIR:DOCKER_DIR:OPTIONS)
CUSTOM_VOLUMES=(
    "/etc/localtime:/etc/localtime:ro"
)



## Select the container name based on the image name
CONTAINER_NAME="${IMAGE_NAME##*/}"
# If the container name is already in use, append a unique (incremental) numerical suffix
if docker container list --all --format "{{.Names}}" | grep -qi "${CONTAINER_NAME}"; then
    CONTAINER_NAME="${CONTAINER_NAME}1"
    while docker container list --all --format "{{.Names}}" | grep -qi "${CONTAINER_NAME}"; do
        CONTAINER_NAME="${CONTAINER_NAME%?}$((${CONTAINER_NAME: -1} + 1))"
    done
fi
DOCKER_RUN_OPTS="--name ${CONTAINER_NAME} ${DOCKER_RUN_OPTS}"

## Parse volumes and environment variables
while getopts ":v:e:" opt; do
    case "${opt}" in
        v) CUSTOM_VOLUMES+=("${OPTARG}") ;;
        e) CUSTOM_ENVS+=("${OPTARG}") ;;
        *)
            echo >&2 "Usage: ${0} [-v HOST_DIR:DOCKER_DIR:OPTIONS] [-e ENV=VALUE] [TAG] [CMD]"
            exit 2
            ;;
    esac
done
shift "$((OPTIND - 1))"

## GPU
if [[ "${ENABLE_GPU,,}" = true ]]; then
    check_nvidia_gpu() {
        if [[ -n "${ENABLE_GPU_FORCE_NVIDIA}" ]]; then
            if [[ "${ENABLE_GPU_FORCE_NVIDIA,,}" = true ]]; then
                echo "INFO: NVIDIA GPU is force-enabled via \`ENABLE_GPU_FORCE_NVIDIA=true\`."
                return 0 # NVIDIA GPU is force-enabled
            else
                echo "INFO: NVIDIA GPU is force-disabled via \`ENABLE_GPU_FORCE_NVIDIA=false\`."
                return 1 # NVIDIA GPU is force-disabled
            fi
        elif ! lshw -C display 2>/dev/null | grep -qi "vendor.*nvidia"; then
            return 1 # NVIDIA GPU is not present
        elif [[ ! -x "$(command -v nvidia-smi)" ]]; then
            echo >&2 -e "\e[33mWARNING: NVIDIA GPU is detected, but its functionality cannot be verified. This container will not be able to use the GPU. Please install nvidia-utils on the host system or force-enable NVIDIA GPU via \`ENABLE_GPU_FORCE_NVIDIA=true\` environment variable.\e[0m"
            return 1 # NVIDIA GPU is present but nvidia-utils not installed
        elif ! nvidia-smi -L &>/dev/null; then
            echo >&2 -e "\e[33mWARNING: NVIDIA GPU is detected, but it does not seem to be working properly. This container will not be able to use the GPU. Please ensure the NVIDIA drivers are properly installed on the host system.\e[0m"
            return 1 # NVIDIA GPU is present but is not working properly
        else
            return 0 # NVIDIA GPU is present and appears to be working
        fi
    }
    if check_nvidia_gpu; then
        # Enable GPU either via NVIDIA Container Toolkit or NVIDIA Docker (depending on Docker version)
        if dpkg --compare-versions "$(docker version --format '{{.Server.Version}}')" gt "19.3"; then
            GPU_OPT="--gpus all"
        else
            GPU_OPT="--runtime nvidia"
        fi
        GPU_ENVS=(
            NVIDIA_VISIBLE_DEVICES="all"
            NVIDIA_DRIVER_CAPABILITIES="all"
        )
    elif [[ $(getent group video) ]]; then
        GPU_OPT="--device=/dev/dri:/dev/dri --group-add video"
    else
        GPU_OPT="--device=/dev/dri:/dev/dri"
    fi
fi

## GUI
if [[ "${ENABLE_GUI,,}" = true ]]; then
    # To enable GUI, make sure processes in the container can connect to the x server
    XAUTH=/tmp/.docker.xauth
    if [ ! -f ${XAUTH} ]; then
        touch ${XAUTH}
        chmod a+r ${XAUTH}

        XAUTH_LIST=$(xauth nlist "${DISPLAY}")
        if [ -n "${XAUTH_LIST}" ]; then
            # shellcheck disable=SC2001
            XAUTH_LIST=$(sed -e 's/^..../ffff/' <<<"${XAUTH_LIST}")
            echo "${XAUTH_LIST}" | xauth -f ${XAUTH} nmerge -
        fi
    fi
    # GUI-enabling volumes
    GUI_VOLUMES=(
        "${XAUTH}:${XAUTH}"
        "/tmp/.X11-unix:/tmp/.X11-unix"
        "/dev/input:/dev/input"
    )
    # GUI-enabling environment variables
    GUI_ENVS=(
        DISPLAY="${DISPLAY}"
        XAUTHORITY="${XAUTH}"
    )
fi

## Run the container
DOCKER_RUN_CMD=(
    docker run
    "${DOCKER_RUN_OPTS}"
    "${GPU_OPT}"
    "${GPU_ENVS[@]/#/"--env "}"
    "${GUI_VOLUMES[@]/#/"--volume "}"
    "${GUI_ENVS[@]/#/"--env "}"
    "${CUSTOM_VOLUMES[@]/#/"--volume "}"
    "${CUSTOM_ENVS[@]/#/"--env "}"
    "${IMAGE_NAME}"
    "${CMD}"
)
echo -e "\033[1;30m${DOCKER_RUN_CMD[*]}\033[0m" | xargs
# shellcheck disable=SC2048
exec ${DOCKER_RUN_CMD[*]}
