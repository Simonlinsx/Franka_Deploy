#!/usr/bin/env bash

# Resolve the one pure-Python dependency needed by the official RH56 XLS
# mapper without adding a complete Conda environment to ROS Python 3.10.
resolve_dexgrasp_xlrd_site() {
  if /usr/bin/python3 -c 'import xlrd' >/dev/null 2>&1; then
    return 0
  fi

  local default_site
  local candidate
  local imported
  default_site="/home/qiaoguanren/anaconda3/pkgs/xlrd-2.0.1-pyhd3eb1b0_0/site-packages"
  candidate="${DEXGRASP_XLRD_SITE:-$default_site}"
  if [[ ! -f "$candidate/xlrd/__init__.py" ]]; then
    echo "[dependency][fatal] xlrd is unavailable in system Python and DEXGRASP_XLRD_SITE is invalid: $candidate" >&2
    return 1
  fi
  candidate="$(cd "$candidate" && pwd -P)"

  imported="$({ PYTHONPATH="$candidate${PYTHONPATH:+:$PYTHONPATH}" /usr/bin/python3 -c 'import pathlib, xlrd; print(pathlib.Path(xlrd.__file__).resolve())'; } 2>/dev/null)" || {
    echo "[dependency][fatal] cannot import xlrd from isolated site: $candidate" >&2
    return 1
  }
  case "$imported" in
    "$candidate"/xlrd/*) ;;
    *)
      echo "[dependency][fatal] xlrd resolved outside the selected isolated site: $imported" >&2
      return 1
      ;;
  esac
  export PYTHONPATH="$candidate${PYTHONPATH:+:$PYTHONPATH}"
  echo "[dependency] isolated xlrd site=$candidate" >&2
}
