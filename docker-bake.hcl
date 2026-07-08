# Build matrix for image creation. Compose is intentionally kept for runtime
# commands because it owns GPU flags, mounts, and shared memory.
variable "PYTORCH_VERSION" {
  default = "2.12.1"
}

variable "TORCH_CUDA_ARCH_LIST" {
  default = "8.6;8.9;12.0"
}

variable "MAX_JOBS" {
  default = "8"
}

variable "GS_IMAGE_PREFIX" {
  default = "gaussian-splatting"
}

group "default" {
  targets = ["cu130"]
}

# Only lanes that have been built and smoke-tested on this workspace belong in
# this group. cu126/cu132 stay in "all" until they pass the same checks.
group "verified" {
  targets = ["cu130"]
}

group "all" {
  targets = ["cu126", "cu130", "cu132"]
}

# All targets use inline cache. Docker Desktop repeatedly produced local-cache
# ref-lock errors when exporting back into the same type=local cache directory.
target "cu126" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  platforms  = ["linux/amd64"]
  tags       = ["${GS_IMAGE_PREFIX}:pytorch-${PYTORCH_VERSION}-cu126"]
  args = {
    PYTORCH_IMAGE       = "pytorch/pytorch:${PYTORCH_VERSION}-cuda12.6-cudnn9-devel"
    TORCH_CUDA_ARCH_LIST = "${TORCH_CUDA_ARCH_LIST}"
    MAX_JOBS            = "${MAX_JOBS}"
  }
  cache-to   = ["type=inline"]
}

target "cu130" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  platforms  = ["linux/amd64"]
  tags       = ["${GS_IMAGE_PREFIX}:pytorch-${PYTORCH_VERSION}-cu130"]
  args = {
    PYTORCH_IMAGE       = "pytorch/pytorch:${PYTORCH_VERSION}-cuda13.0-cudnn9-devel"
    TORCH_CUDA_ARCH_LIST = "${TORCH_CUDA_ARCH_LIST}"
    MAX_JOBS            = "${MAX_JOBS}"
  }
  cache-to   = ["type=inline"]
}

target "cu132" {
  context    = "."
  dockerfile = "docker/Dockerfile"
  platforms  = ["linux/amd64"]
  tags       = ["${GS_IMAGE_PREFIX}:pytorch-${PYTORCH_VERSION}-cu132"]
  args = {
    PYTORCH_IMAGE       = "pytorch/pytorch:${PYTORCH_VERSION}-cuda13.2-cudnn9-devel"
    TORCH_CUDA_ARCH_LIST = "${TORCH_CUDA_ARCH_LIST}"
    MAX_JOBS            = "${MAX_JOBS}"
  }
  cache-to   = ["type=inline"]
}
