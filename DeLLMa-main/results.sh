#!/usr/bin/env bash
# Auto-evaluate all available result folders.
# By default reads DeLLMa-main/results, which may contain per-model subfolders.
#
# Usage:
#   ./results.sh
#   ./results.sh --dir /path/to/results
#   ./results.sh --model-tag Llama-3.1-8B-Instruct --agent trader
#   ./results.sh --model-tag Llama-3.1-8B-Instruct --agent trader --max-choices 3
#   ./results.sh --model-tag Llama-3.1-8B-Instruct --agent trader --summary-out results/summary.txt
#   RESULTS=/path/to/results ./results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS="${RESULTS:-$SCRIPT_DIR/results}"
PY="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage:
  ./results.sh
  ./results.sh --dir /path/to/results
  ./results.sh --model-tag <folder-name> [--agent <farmer|trader|powergrid|seir>]
  ./results.sh --model-tag <folder-name> --agent <farmer|trader|powergrid|seir> [--max-choices <X>] [--summary-out <path>]

Notes:
  - If omitted, results dir defaults to ./results
  - Environment variable RESULTS is also supported
  - --model-tag limits evaluation to one model folder under results/
  - --agent limits evaluation to one application
  - --max-choices must match main.py/run.sh when inference used a limited pool
  - --summary-out writes a clean text summary (relative paths resolve from DeLLMa-main/)
  - If --model-tag and --agent are both set, default summary path is:
      results/<model-tag>/<domain>/<agent>_eval_summary.txt
    (trader -> stocks/, powergrid -> powergrid/, seir -> seir/, farmer -> agriculture/2021/)
EOF
}

agent_domain_rel_path() {
  case "${1:-}" in
    farmer) echo "agriculture/2021" ;;
    trader) echo "stocks" ;;
    powergrid) echo "powergrid" ;;
    seir) echo "seir" ;;
    *) echo "" ;;
  esac
}

MODEL_TAG=""
AGENT_FILTER=""
SUMMARY_OUT=""
MAX_CHOICES=""
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --dir)
      RESULTS="${2:-}"
      if [[ -z "${RESULTS}" ]]; then
        echo "Missing value for --dir" >&2
        usage >&2
        exit 1
      fi
      shift 2
      ;;
    --model-tag)
      MODEL_TAG="${2:-}"
      if [[ -z "${MODEL_TAG}" ]]; then
        echo "Missing value for --model-tag" >&2
        usage >&2
        exit 1
      fi
      shift 2
      ;;
    --agent)
      AGENT_FILTER="${2:-}"
      case "${AGENT_FILTER}" in
        farmer|trader|powergrid|seir) ;;
        *)
          echo "Invalid --agent value: ${AGENT_FILTER}. Use farmer|trader|powergrid|seir." >&2
          usage >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --summary-out)
      SUMMARY_OUT="${2:-}"
      if [[ -z "${SUMMARY_OUT}" ]]; then
        echo "Missing value for --summary-out" >&2
        usage >&2
        exit 1
      fi
      shift 2
      ;;
    --max-choices)
      MAX_CHOICES="${2:-}"
      if [[ -z "${MAX_CHOICES}" ]]; then
        echo "Missing value for --max-choices" >&2
        usage >&2
        exit 1
      fi
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "Unexpected extra arguments: $*" >&2
  usage >&2
  exit 1
fi

if [[ ! -d "$RESULTS" ]]; then
  echo "Results directory does not exist: $RESULTS" >&2
  exit 1
fi

if [[ -z "$SUMMARY_OUT" && -n "$MODEL_TAG" && -n "$AGENT_FILTER" ]]; then
  domain_rel="$(agent_domain_rel_path "$AGENT_FILTER")"
  SUMMARY_OUT="$RESULTS/$MODEL_TAG/$domain_rel/${AGENT_FILTER}_eval_summary.txt"
fi

if [[ -n "$SUMMARY_OUT" && "$SUMMARY_OUT" != /* ]]; then
  SUMMARY_OUT="$SCRIPT_DIR/$SUMMARY_OUT"
fi

if [[ -n "$SUMMARY_OUT" ]]; then
  mkdir -p "$(dirname "$SUMMARY_OUT")"
  summary_title="Evaluation Summary"
  case "${AGENT_FILTER}" in
    trader) summary_title="Trader Evaluation Summary" ;;
    farmer) summary_title="Farmer Evaluation Summary" ;;
    powergrid) summary_title="Powergrid Evaluation Summary" ;;
    seir) summary_title="SEIR Evaluation Summary" ;;
  esac
  {
    echo "$summary_title"
    [[ -n "$MODEL_TAG" ]] && echo "model: $MODEL_TAG"
    [[ -n "$AGENT_FILTER" ]] && echo "agent: $AGENT_FILTER"
    [[ -n "$MAX_CHOICES" ]] && echo "max_choices: $MAX_CHOICES"
    echo "generated_at: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo
    printf '%-60s  %7s  %7s\n' "method" "all_acc" "all_opt"
    printf '%s\n' "---------------------------------------------------------------------------------------------------"
  } > "$SUMMARY_OUT"
fi

is_result_root() {
  local p="$1"
  [[ -d "$p/agriculture" || -d "$p/stocks" || -d "$p/powergrid" || -d "$p/seir" ]]
}

is_domain_root() {
  local p="$1"
  # trader/powergrid direct layouts
  if [[ -d "$p/zero-shot" || -d "$p/self-consistency" || -d "$p/cot" || -d "$p/dellma" || -d "$p/dmuu" ]]; then
    return 0
  fi
  # farmer direct layout includes year subdirs (e.g., 2021/zero-shot)
  for y in "$p"/*; do
    [[ -d "$y" ]] || continue
    if [[ -d "$y/zero-shot" || -d "$y/self-consistency" || -d "$y/cot" || -d "$y/dellma" || -d "$y/dmuu" ]]; then
      return 0
    fi
  done
  return 1
}

run_eval() {
  local label="$1"
  shift
  echo "==> $label"
  local method_name cfg_name method_field
  method_name="$(printf "%s" "$label" | sed -E 's/.* mode=([^ ]+).*/\1/')"
  cfg_name="$(printf "%s" "$label" | sed -nE 's/.* cfg=([^ ]+).*/\1/p')"
  method_field="$method_name"
  if [[ -n "$cfg_name" ]]; then
    pretty_cfg="$cfg_name"
    pretty_cfg="${pretty_cfg/_minibatch_size=/,minibatch_size=}"
    pretty_cfg="${pretty_cfg/_overlap_pct=/,overlap=}"
    method_field="${method_name}(${pretty_cfg})"
  fi
  local tmp
  tmp="$(mktemp)"
  if ! "$PY" evaluate_dellma.py "$@" >"$tmp" 2>&1; then
    echo "    Status: FAILED"
    sed 's/^/    /' "$tmp"
    if [[ -n "$SUMMARY_OUT" ]]; then
      printf '%-60s  %7s  %7s\n' "$method_field" "-" "-" >> "$SUMMARY_OUT"
    fi
    rm -f "$tmp"
    return 1
  fi

  local all_acc all_opt all_regret all_ig_regret err_count
  all_acc="$(awk '$1=="All-Acc"{print $2; exit}' "$tmp")"
  all_opt="$(awk '/^All-Opt \(%\)/{print $3; exit}' "$tmp")"
  all_regret="$(awk '/^All-Regret /{print $NF; exit}' "$tmp")"
  all_ig_regret="$(awk '/^All-IG-Regret/{print $NF; exit}' "$tmp")"
  err_count="$(awk '/^error reading response file /{c++} END{print c+0}' "$tmp")"

  if [[ -n "$all_acc" ]]; then
    echo "    All-Acc: $all_acc"
  fi
  if [[ -n "$all_opt" ]]; then
    echo "    All-Opt (%): $all_opt"
  elif [[ -n "$all_regret" ]]; then
    echo "    All-Regret: $all_regret"
  elif [[ -n "$all_ig_regret" ]]; then
    echo "    All-IG-Regret: $all_ig_regret"
  fi
  if [[ "$err_count" -gt 0 ]]; then
    echo "    Warnings: $err_count unreadable response files"
  fi

  if [[ "$VERBOSE" -eq 1 ]]; then
    echo "    --- raw evaluator output ---"
    sed 's/^/    /' "$tmp"
    echo "    --- end raw output ---"
  fi

  if [[ -n "$SUMMARY_OUT" ]]; then
    summary_opt="${all_opt:-}"
    if [[ -z "$summary_opt" && -n "${all_regret:-}" ]]; then
      summary_opt="$all_regret"
    fi
    if [[ -z "$summary_opt" && -n "${all_ig_regret:-}" ]]; then
      summary_opt="$all_ig_regret"
    fi
    printf '%-60s  %7s  %7s\n' \
      "$method_field" \
      "${all_acc:-}" \
      "${summary_opt:-}" >> "$SUMMARY_OUT"
  fi

  rm -f "$tmp"
}

evaluate_agent_modes() {
  local root="$1"
  local tag="$2"
  local agent="$3"

  local domain=""
  local year_path=""
  local extra_args=()
  case "$agent" in
    farmer)
      domain="agriculture"
      year_path="2021"
      extra_args=(--agent_name farmer --year 2021)
      ;;
    trader)
      domain="stocks"
      year_path=""
      extra_args=(--agent_name trader)
      ;;
    powergrid)
      domain="powergrid"
      year_path=""
      extra_args=(--agent_name powergrid)
      ;;
    seir)
      domain="seir"
      year_path=""
      extra_args=(--agent_name seir)
      ;;
    *)
      return
      ;;
  esac
  if [[ -n "$MAX_CHOICES" ]]; then
    extra_args+=(--max_choices "$MAX_CHOICES")
  fi

  local base_prefix="$root/$domain"
  if [[ -n "$year_path" ]]; then
    base_prefix="$base_prefix/$year_path"
  fi

  local mode
  for mode in zero-shot self-consistency cot; do
    if [[ -d "$base_prefix/$mode" ]]; then
      if [[ "$mode" == "cot" ]]; then
        # Some runs save CoT as response.json; evaluator expects response_0.json.
        for resp_dir in "$base_prefix/$mode"/*/response; do
          [[ -d "$resp_dir" ]] || continue
          if [[ -f "$resp_dir/response.json" && ! -f "$resp_dir/response_0.json" ]]; then
            cp "$resp_dir/response.json" "$resp_dir/response_0.json"
          fi
        done
      fi
      run_eval \
        "model=${tag:-<none>} agent=$agent mode=$mode" \
        --results_path "$root" --model_tag "" \
        "${extra_args[@]}" \
        --pref_enum_mode "$mode"
    fi
  done

  local rank_root="dellma"
  if [[ ! -d "$base_prefix/$rank_root" && -d "$base_prefix/dmuu" ]]; then
    rank_root="dmuu"
  fi

  if [[ -d "$base_prefix/$rank_root/rank" ]]; then
    run_eval \
      "model=${tag:-<none>} agent=$agent mode=rank" \
      --results_path "$root" --model_tag "" \
      "${extra_args[@]}" \
      --pref_enum_mode rank
  fi

  if [[ -d "$base_prefix/$rank_root/rank-minibatch" ]]; then
    local cfg
    for cfg in "$base_prefix"/"$rank_root"/rank-minibatch/*; do
      [[ -d "$cfg" ]] || continue
      local cfg_name
      cfg_name="$(basename "$cfg")"
      if [[ "$cfg_name" =~ sample_size=([0-9]+)_minibatch_size=([0-9]+)_overlap_pct=([0-9]+) ]]; then
        local sample_size="${BASH_REMATCH[1]}"
        local minibatch_size="${BASH_REMATCH[2]}"
        local overlap_pct_int="${BASH_REMATCH[3]}"
        local overlap_pct
        overlap_pct="$(awk "BEGIN { print ${overlap_pct_int}/100 }")"
        run_eval \
          "model=${tag:-<none>} agent=$agent mode=rank-minibatch cfg=$cfg_name" \
          --results_path "$root" --model_tag "" \
          "${extra_args[@]}" \
          --pref_enum_mode rank-minibatch \
          --sample_size "$sample_size" \
          --minibatch_size "$minibatch_size" \
          --overlap_pct "$overlap_pct"
      fi
    done
  fi
}

echo "Using results directory: $RESULTS"

roots=()
tags=()
root_modes=()

if is_result_root "$RESULTS"; then
  if [[ -z "$MODEL_TAG" || "$MODEL_TAG" == "$(basename "$RESULTS")" ]]; then
    roots+=("$RESULTS")
    tags+=("")
    root_modes+=("full")
  fi
elif is_domain_root "$RESULTS"; then
  roots+=("$RESULTS")
  tags+=("")
  root_modes+=("domain")
fi

for d in "$RESULTS"/*; do
  [[ -d "$d" ]] || continue
  if [[ -n "$MODEL_TAG" && "$(basename "$d")" != "$MODEL_TAG" ]]; then
    continue
  fi
  if is_result_root "$d"; then
    roots+=("$d")
    tags+=("$(basename "$d")")
    root_modes+=("full")
  elif is_domain_root "$d"; then
    roots+=("$d")
    tags+=("$(basename "$d")")
    root_modes+=("domain")
  fi
done

if [[ "${#roots[@]}" -eq 0 ]]; then
  echo "No valid result roots found under: $RESULTS"
  if [[ -n "$MODEL_TAG" ]]; then
    echo "Tried model tag filter: $MODEL_TAG"
  fi
  exit 1
fi

for idx in "${!roots[@]}"; do
  root="${roots[$idx]}"
  tag="${tags[$idx]}"
  mode="${root_modes[$idx]}"
  echo
  echo "######## Evaluating root: $root ${tag:+(model tag: $tag)} ########"
  if [[ "$mode" == "domain" ]]; then
    base="$(basename "$root")"
    case "$base" in
      stocks)
        if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "trader" ]]; then
          evaluate_agent_modes "$root/.." "$tag" trader
        fi
        ;;
      powergrid)
        if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "powergrid" ]]; then
          evaluate_agent_modes "$root/.." "$tag" powergrid
        fi
        ;;
      seir)
        if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "seir" ]]; then
          evaluate_agent_modes "$root/.." "$tag" seir
        fi
        ;;
      agriculture)
        if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "farmer" ]]; then
          evaluate_agent_modes "$root/.." "$tag" farmer
        fi
        ;;
      *)
        # try all agents relative to parent for robustness
        if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "farmer" ]]; then
          evaluate_agent_modes "$root/.." "$tag" farmer
        fi
        if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "trader" ]]; then
          evaluate_agent_modes "$root/.." "$tag" trader
        fi
        if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "powergrid" ]]; then
          evaluate_agent_modes "$root/.." "$tag" powergrid
        fi
        if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "seir" ]]; then
          evaluate_agent_modes "$root/.." "$tag" seir
        fi
        ;;
    esac
  else
    if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "farmer" ]]; then
      evaluate_agent_modes "$root" "$tag" farmer
    fi
    if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "trader" ]]; then
      evaluate_agent_modes "$root" "$tag" trader
    fi
    if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "powergrid" ]]; then
      evaluate_agent_modes "$root" "$tag" powergrid
    fi
    if [[ -z "$AGENT_FILTER" || "$AGENT_FILTER" == "seir" ]]; then
      evaluate_agent_modes "$root" "$tag" seir
    fi
  fi
done

echo
echo "All available evaluations completed."
if [[ -n "$SUMMARY_OUT" ]]; then
  echo "Summary saved to: $SUMMARY_OUT"
fi
