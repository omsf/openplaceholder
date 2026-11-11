#!/usr/bin/env bash
# Bring up a local kind cluster running Prefect and KubeRay, and load the image.
set -euo pipefail

CLUSTER=oph
KUBERAY=v1.7.2
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
RUN=${OPH_KUBERAY_RUN:-$HOME/.local/state/oph-kuberay}

mkdir -p "$RUN"

# kind and kustomize both stage GB-sized files, and /tmp may be a small tmpfs
export TMPDIR=${OPH_TMPDIR:-$RUN/.tmp}
mkdir -p "$TMPDIR"

# the kind node already carries /dev/nvidia* but not the driver's libraries
CUDA_MAX=13.2
if command -v nvidia-smi >/dev/null 2>&1; then
  detected=$(nvidia-smi --version | awk -F: '/CUDA Version/ {gsub(/ /, "", $2); print $2}')
  if [ -n "$detected" ]; then
    CUDA_MAX=$detected
  fi
  mkdir -p "$RUN/.nvidia"
  (cd /usr/lib/x86_64-linux-gnu \
   && cp -an libcuda.so* libnvidia-ml.so* libnvidia-nvvm.so* \
        libnvidia-ptxjitcompiler.so* libnvidia-gpucomp.so* libnvidia-tls.so* \
        /usr/bin/nvidia-smi "$RUN/.nvidia"/) || true
fi

# a conda env only runs at the path it was built at, so mount it at that path
OF3=${OPH_OF3_HOME:-$HOME/software/openfold-3}
OF3_CACHE=${OPENFOLD_CACHE:-$HOME/.openfold3}
OF3_MOUNTS=""
if [ -x "$OF3/.pixi/envs/openfold3-cuda13/bin/run_openfold" ]; then
  OF3_MOUNTS="
      - hostPath: $OF3
        containerPath: $OF3
        readOnly: true
      - hostPath: $OF3_CACHE
        containerPath: $OF3_CACHE
        readOnly: true"
  # what `pixi run` would export, for the generator's subprocess only
  mkdir -p "$RUN/.of3"
  cat > "$RUN/.of3/run_openfold" <<SHIM
#!/bin/sh
export CONDA_PREFIX=$OF3/.pixi/envs/openfold3-cuda13
export CUDA_HOME=\$CONDA_PREFIX
export PATH=\$CONDA_PREFIX/bin:\$PATH
exec run_openfold "\$@"
SHIM
  chmod +x "$RUN/.of3/run_openfold"

  # extraMounts only take effect at creation
  if kind get clusters | grep -qx "$CLUSTER" \
     && ! docker exec "$CLUSTER-control-plane" test -d "$OF3" 2>/dev/null; then
    echo "cluster '$CLUSTER' predates the openfold3 mount; recreating it"
    kind delete cluster --name "$CLUSTER"
  fi
else
  echo "no openfold3 env under $OF3: the generator stage will not run" >&2
  # the manifest still needs paths that parse as absolute
  OF3=/data/.of3-absent
  OF3_CACHE=/data/.of3-absent
fi

if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --config - <<YAML
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraMounts:
      - hostPath: $RUN
        containerPath: /data$OF3_MOUNTS
YAML
fi

docker build --build-arg "CUDA_MAX=$CUDA_MAX" -t oph-kuberay:dev -f "$HERE/Dockerfile" "$REPO"
kind load docker-image oph-kuberay:dev --name "$CLUSTER"

kubectl apply --server-side -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=$KUBERAY"
kubectl rollout status deploy/kuberay-operator --timeout=300s

sed "s#__OF3_HOME__#$OF3#g; s#__OF3_CACHE__#$OF3_CACHE#g" "$HERE/cluster.yaml" \
  | kubectl apply -f -
kubectl rollout status deploy/prefect-server --timeout=600s
kubectl wait --for=condition=Ready pod -l ray.io/cluster="$CLUSTER" --timeout=600s

echo
echo "cluster up. run the pipeline with:"
echo "  kubectl delete job oph-tyk2 --ignore-not-found"
echo "  kubectl apply -f $HERE/tyk2.yaml"
echo "  kubectl logs -f job/oph-tyk2"
