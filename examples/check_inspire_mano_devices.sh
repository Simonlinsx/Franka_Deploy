#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
camera_only=false
gpu_check=false
while (($#)); do
  case "$1" in
    --camera-only)
      camera_only=true
      ;;
    --gpu)
      gpu_check=true
      ;;
    *)
      echo "Usage: $0 [--camera-only] [--gpu]" >&2
      exit 64
      ;;
  esac
  shift
done

camera_found=false
rh56_usb_found=false

echo "USB devices relevant to this pipeline:"
for device in /sys/bus/usb/devices/*; do
  [[ -r "$device/idVendor" && -r "$device/idProduct" ]] || continue
  read -r vendor < "$device/idVendor"
  read -r product < "$device/idProduct"
  case "$vendor:$product" in
    8086:*)
      product_name=""
      serial=""
      [[ -r "$device/product" ]] && read -r product_name < "$device/product"
      [[ -r "$device/serial" ]] && read -r serial < "$device/serial"
      if [[ "$product_name" == *RealSense* ]]; then
        camera_found=true
        echo "  RealSense: $vendor:$product $product_name serial=$serial"
      fi
      ;;
    1a86:7523)
      rh56_usb_found=true
      driver="none"
      for interface in "$device":*; do
        [[ -L "$interface/driver" ]] || continue
        driver="$(basename "$(readlink -f "$interface/driver")")"
        break
      done
      echo "  RH56 USB adapter: $vendor:$product driver=$driver sysfs=$(basename "$device")"
      ;;
  esac
done

if command -v rs-enumerate-devices >/dev/null 2>&1; then
  echo
  echo "RealSense SDK summary:"
  rs-enumerate-devices -s || true
fi

gpu_ok=true
if [[ "$gpu_check" == true ]]; then
  echo
  echo "CUDA health:"
  pipeline_python="$ROOT_DIR/examples/inspire_mano_pipeline/.venv/bin/python"
  if [[ ! -x "$pipeline_python" ]]; then
    echo "  FAIL: pipeline environment is missing; run examples/setup_inspire_mano_env.sh." >&2
    gpu_ok=false
  elif "$pipeline_python" -c 'import torch; ok=torch.cuda.is_available() and torch.cuda.device_count() > 0; print(f"  torch={torch.__version__} available={torch.cuda.is_available()} devices={torch.cuda.device_count()}"); print(f"  device0={torch.cuda.get_device_name(0)}" if ok else "  device0=unavailable"); raise SystemExit(0 if ok else 1)'; then
    echo "  PASS: CUDA is available to the pipeline environment."
  else
    echo "  FAIL: CUDA is unavailable to the pipeline environment." >&2
    echo "  If nvidia-smi sees the GPU and no CUDA jobs are running, recover nvidia_uvm with:" >&2
    echo "    sudo modprobe -r nvidia_uvm" >&2
    echo "    sudo modprobe nvidia_uvm" >&2
    gpu_ok=false
  fi
fi

serial_nodes=()
while IFS= read -r -d '' node; do
  serial_nodes+=("$node")
done < <(find /dev/serial/by-id -maxdepth 1 -type l -print0 2>/dev/null || true)

echo
if ((${#serial_nodes[@]})); then
  echo "Persistent serial nodes:"
  for node in "${serial_nodes[@]}"; do
    printf '  %s -> %s\n' "$node" "$(readlink -f "$node")"
  done
else
  echo "Persistent serial nodes: none"
fi

brltty_unit_state="$(systemctl is-enabled brltty-udev.service 2>/dev/null || true)"
if [[ "$brltty_unit_state" == masked* ]]; then
  echo "BRLTTY udev service: $brltty_unit_state"
elif systemctl is-active --quiet brltty-udev.service 2>/dev/null; then
  echo "BRLTTY udev service: active"
else
  echo "BRLTTY udev service: inactive/unknown"
fi

if [[ "$camera_found" != true ]]; then
  echo "FAIL: no Intel RealSense camera was found in sysfs." >&2
  exit 2
fi
if [[ "$gpu_ok" != true ]]; then
  exit 5
fi
if [[ "$camera_only" == true ]]; then
  if [[ "$gpu_check" == true ]]; then
    echo "PASS: RealSense and CUDA are present (camera-only + GPU check)."
  else
    echo "PASS: RealSense is present (camera-only check)."
  fi
  exit 0
fi
if [[ "$rh56_usb_found" != true ]]; then
  echo "FAIL: RH56 CH340 USB adapter 1a86:7523 was not found." >&2
  exit 3
fi
if ((${#serial_nodes[@]} == 0)); then
  echo "FAIL: RH56 USB exists but has no /dev/serial/by-id node." >&2
  echo "If BRLTTY is not needed: runtime-mask brltty-udev, reload ch341, then replug USB." >&2
  echo "  sudo systemctl mask --runtime --now brltty-udev.service" >&2
  echo "  sudo modprobe -r ch341; sudo modprobe ch341" >&2
  exit 4
fi

echo "PASS: RealSense and RH56 serial node are both present."
