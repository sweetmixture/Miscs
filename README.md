```
#!/usr/bin/env bash
#
# setup-mc.sh
#   - jq 설치 (apt)
#   - MinIO Client(mc) 설치 (공식 바이너리)
#   - ~/.mc/config.json 자동 생성 (MinIO alias + AWS S3 alias)
#
# 사용법:
#   MINIO_URL=http://192.168.0.10:9000 \
#   MINIO_ACCESS_KEY=xxx MINIO_SECRET_KEY=yyy \
#   S3_ACCESS_KEY=AKIA... S3_SECRET_KEY=zzz \
#   ./setup-mc.sh
#
# 환경변수를 안 주면 실행 중 프롬프트로 물어봅니다.

set -euo pipefail

# ── 설정값 (환경변수로 덮어쓰기 가능) ────────────────────────────
MC_BIN_DIR="${MC_BIN_DIR:-/usr/local/bin}"
MC_BIN_NAME="${MC_BIN_NAME:-mc}"          # midnight commander와 겹치면 mcli 로 변경
MC_CONFIG_DIR="${MC_CONFIG_DIR:-$HOME/.mc}"
CONFIG_FILE="$MC_CONFIG_DIR/config.json"

MINIO_ALIAS="${MINIO_ALIAS:-myminio}"
MINIO_URL="${MINIO_URL:-}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-}"

S3_ALIAS="${S3_ALIAS:-s3}"
S3_URL="${S3_URL:-https://s3.amazonaws.com}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-}"
S3_SECRET_KEY="${S3_SECRET_KEY:-}"

# ── 유틸 ────────────────────────────────────────────────────────
log()  { printf '\033[1;32m[+]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

SUDO=""
if [[ $EUID -ne 0 ]]; then
  command -v sudo >/dev/null 2>&1 || die "root가 아니고 sudo도 없습니다."
  SUDO="sudo"
fi

ask() {                       # ask <변수명> <프롬프트> [secret]
  local __var=$1 __prompt=$2 __secret=${3:-} __val
  if [[ -n "${!__var}" ]]; then return 0; fi
  if [[ ! -t 0 ]]; then die "$__var 값이 없습니다. 환경변수로 지정하세요."; fi
  if [[ -n "$__secret" ]]; then
    read -r -s -p "$__prompt: " __val; echo
  else
    read -r -p "$__prompt: " __val
  fi
  [[ -n "$__val" ]] || die "$__var 는 필수입니다."
  printf -v "$__var" '%s' "$__val"
}

# ── 1. jq 설치 ──────────────────────────────────────────────────
if command -v jq >/dev/null 2>&1; then
  log "jq 이미 설치됨 ($(jq --version))"
else
  log "jq 설치 중..."
  $SUDO apt-get update -qq
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq jq curl ca-certificates
  log "jq 설치 완료 ($(jq --version))"
fi

# ── 2. mc 설치 ──────────────────────────────────────────────────
# Midnight Commander 충돌 확인
if command -v mc >/dev/null 2>&1 && ! mc --version 2>/dev/null | grep -qi minio; then
  warn "기존 'mc'가 MinIO client가 아닙니다 (Midnight Commander로 보임)."
  MC_BIN_NAME="mcli"
  warn "MinIO client를 '$MC_BIN_NAME' 이름으로 설치합니다."
fi

MC="$MC_BIN_DIR/$MC_BIN_NAME"

if [[ -x "$MC" ]] && "$MC" --version 2>/dev/null | grep -qi minio; then
  log "mc 이미 설치됨: $MC"
else
  case "$(uname -m)" in
    x86_64)          MC_ARCH=amd64 ;;
    aarch64|arm64)   MC_ARCH=arm64 ;;
    *) die "지원하지 않는 아키텍처: $(uname -m)" ;;
  esac

  log "mc 다운로드 중 (linux-$MC_ARCH)..."
  TMP_MC="$(mktemp)"
  trap 'rm -f "$TMP_MC"' EXIT
  curl -fsSL -o "$TMP_MC" \
    "https://dl.min.io/client/mc/release/linux-${MC_ARCH}/mc" \
    || die "다운로드 실패 — 네트워크/프록시를 확인하세요."

  chmod +x "$TMP_MC"
  $SUDO install -m 0755 "$TMP_MC" "$MC"
  log "mc 설치 완료: $MC"
fi

"$MC" --version | head -n1

# ── 3. 인증정보 수집 ────────────────────────────────────────────
ask MINIO_URL        "MinIO 엔드포인트 (예: http://192.168.0.10:9000)"
ask MINIO_ACCESS_KEY "MinIO Access Key"
ask MINIO_SECRET_KEY "MinIO Secret Key" secret

if [[ -z "$S3_ACCESS_KEY" || -z "$S3_SECRET_KEY" ]]; then
  if [[ -t 0 ]]; then
    read -r -p "AWS S3 alias도 등록할까요? [y/N]: " _yn
    if [[ "${_yn,,}" == "y" ]]; then
      ask S3_ACCESS_KEY "AWS Access Key ID"
      ask S3_SECRET_KEY "AWS Secret Access Key" secret
    fi
  fi
fi

# ── 4. config.json 생성 ─────────────────────────────────────────
umask 077
mkdir -p "$MC_CONFIG_DIR"
chmod 700 "$MC_CONFIG_DIR"

if [[ -f "$CONFIG_FILE" ]]; then
  BACKUP="$CONFIG_FILE.bak.$(date +%Y%m%d%H%M%S)"
  cp -p "$CONFIG_FILE" "$BACKUP"
  log "기존 설정 백업: $BACKUP"
  BASE_JSON="$(cat "$CONFIG_FILE")"
else
  BASE_JSON='{"version":"10","aliases":{}}'
fi

NEW_JSON="$(
  jq -n \
    --argjson base "$BASE_JSON" \
    --arg ma "$MINIO_ALIAS" --arg mu "$MINIO_URL" \
    --arg mak "$MINIO_ACCESS_KEY" --arg msk "$MINIO_SECRET_KEY" \
    --arg sa "$S3_ALIAS" --arg su "$S3_URL" \
    --arg sak "$S3_ACCESS_KEY" --arg ssk "$S3_SECRET_KEY" '
    ($base | .version = "10" | .aliases //= {})
    | .aliases[$ma] = {
        url: $mu, accessKey: $mak, secretKey: $msk,
        api: "S3v4", path: "auto"
      }
    | if ($sak | length) > 0 then
        .aliases[$sa] = {
          url: $su, accessKey: $sak, secretKey: $ssk,
          api: "S3v4", path: "auto"
        }
      else . end
  '
)" || die "config.json 생성 실패 (jq)"

printf '%s\n' "$NEW_JSON" > "$CONFIG_FILE"
chmod 600 "$CONFIG_FILE"
log "설정 저장: $CONFIG_FILE (600)"

# ── 5. 검증 ─────────────────────────────────────────────────────
log "등록된 alias:"
"$MC" alias list | sed 's/^/    /'

log "연결 테스트: $MINIO_ALIAS"
if "$MC" ls "$MINIO_ALIAS" >/dev/null 2>&1; then
  log "MinIO 연결 정상"
else
  warn "MinIO 연결 실패 — URL/키/방화벽을 확인하세요."
  warn "자체서명 인증서라면: $MC --insecure ls $MINIO_ALIAS"
fi

if [[ -n "$S3_ACCESS_KEY" ]]; then
  log "연결 테스트: $S3_ALIAS"
  "$MC" ls "$S3_ALIAS" >/dev/null 2>&1 \
    && log "S3 연결 정상" \
    || warn "S3 연결 실패 — 키/리전/권한을 확인하세요."
fi

echo
log "완료. 사용 예: $MC ls $MINIO_ALIAS"

```

```
#!/usr/bin/env bash
#
# setup-mc.sh
#   - jq 설치 (apt)
#   - MinIO Client(mc) 설치 (공식 바이너리)
#   - ~/.mc/config.json 자동 생성 (MinIO alias + AWS S3 alias)
#
# 사용법:
#   MINIO_URL=http://192.168.0.10:9000 \
#   MINIO_ACCESS_KEY=xxx MINIO_SECRET_KEY=yyy \
#   S3_ACCESS_KEY=AKIA... S3_SECRET_KEY=zzz \
#   ./setup-mc.sh
#
# 환경변수를 안 주면 실행 중 프롬프트로 물어봅니다.

set -euo pipefail

# ── 설정값 (환경변수로 덮어쓰기 가능) ────────────────────────────
MC_BIN_DIR="${MC_BIN_DIR:-/usr/local/bin}"
MC_BIN_NAME="${MC_BIN_NAME:-mc}"          # midnight commander와 겹치면 mcli 로 변경
MC_CONFIG_DIR="${MC_CONFIG_DIR:-$HOME/.mc}"
CONFIG_FILE="$MC_CONFIG_DIR/config.json"

MINIO_ALIAS="${MINIO_ALIAS:-myminio}"
MINIO_URL="${MINIO_URL:-}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-}"

S3_ALIAS="${S3_ALIAS:-s3}"
S3_URL="${S3_URL:-https://s3.amazonaws.com}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-}"
S3_SECRET_KEY="${S3_SECRET_KEY:-}"

# ── 유틸 ────────────────────────────────────────────────────────
log()  { printf '\033[1;32m[+]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

SUDO=""
if [[ $EUID -ne 0 ]]; then
  command -v sudo >/dev/null 2>&1 || die "root가 아니고 sudo도 없습니다."
  SUDO="sudo"
fi

ask() {                       # ask <변수명> <프롬프트> [secret]
  local __var=$1 __prompt=$2 __secret=${3:-} __val
  if [[ -n "${!__var}" ]]; then return 0; fi
  if [[ ! -t 0 ]]; then die "$__var 값이 없습니다. 환경변수로 지정하세요."; fi
  if [[ -n "$__secret" ]]; then
    read -r -s -p "$__prompt: " __val; echo
  else
    read -r -p "$__prompt: " __val
  fi
  [[ -n "$__val" ]] || die "$__var 는 필수입니다."
  printf -v "$__var" '%s' "$__val"
}

# ── 1. jq 설치 ──────────────────────────────────────────────────
if command -v jq >/dev/null 2>&1; then
  log "jq 이미 설치됨 ($(jq --version))"
else
  log "jq 설치 중..."
  $SUDO apt-get update -qq
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq jq curl ca-certificates
  log "jq 설치 완료 ($(jq --version))"
fi

# ── 2. mc 설치 ──────────────────────────────────────────────────
# Midnight Commander 충돌 확인
if command -v mc >/dev/null 2>&1 && ! mc --version 2>/dev/null | grep -qi minio; then
  warn "기존 'mc'가 MinIO client가 아닙니다 (Midnight Commander로 보임)."
  MC_BIN_NAME="mcli"
  warn "MinIO client를 '$MC_BIN_NAME' 이름으로 설치합니다."
fi

MC="$MC_BIN_DIR/$MC_BIN_NAME"

if [[ -x "$MC" ]] && "$MC" --version 2>/dev/null | grep -qi minio; then
  log "mc 이미 설치됨: $MC"
else
  case "$(uname -m)" in
    x86_64)          MC_ARCH=amd64 ;;
    aarch64|arm64)   MC_ARCH=arm64 ;;
    *) die "지원하지 않는 아키텍처: $(uname -m)" ;;
  esac

  log "mc 다운로드 중 (linux-$MC_ARCH)..."
  TMP_MC="$(mktemp)"
  trap 'rm -f "$TMP_MC"' EXIT
  curl -fsSL -o "$TMP_MC" \
    "https://dl.min.io/client/mc/release/linux-${MC_ARCH}/mc" \
    || die "다운로드 실패 — 네트워크/프록시를 확인하세요."

  chmod +x "$TMP_MC"
  $SUDO install -m 0755 "$TMP_MC" "$MC"
  log "mc 설치 완료: $MC"
fi

"$MC" --version | head -n1

# ── 3. 인증정보 수집 ────────────────────────────────────────────
ask MINIO_URL        "MinIO 엔드포인트 (예: http://192.168.0.10:9000)"
ask MINIO_ACCESS_KEY "MinIO Access Key"
ask MINIO_SECRET_KEY "MinIO Secret Key" secret

if [[ -z "$S3_ACCESS_KEY" || -z "$S3_SECRET_KEY" ]]; then
  if [[ -t 0 ]]; then
    read -r -p "AWS S3 alias도 등록할까요? [y/N]: " _yn
    if [[ "${_yn,,}" == "y" ]]; then
      ask S3_ACCESS_KEY "AWS Access Key ID"
      ask S3_SECRET_KEY "AWS Secret Access Key" secret
    fi
  fi
fi

# ── 4. config.json 생성 ─────────────────────────────────────────
umask 077
mkdir -p "$MC_CONFIG_DIR"
chmod 700 "$MC_CONFIG_DIR"

if [[ -f "$CONFIG_FILE" ]]; then
  BACKUP="$CONFIG_FILE.bak.$(date +%Y%m%d%H%M%S)"
  cp -p "$CONFIG_FILE" "$BACKUP"
  log "기존 설정 백업: $BACKUP"
  BASE_JSON="$(cat "$CONFIG_FILE")"
else
  BASE_JSON='{"version":"10","aliases":{}}'
fi

NEW_JSON="$(
  jq -n \
    --argjson base "$BASE_JSON" \
    --arg ma "$MINIO_ALIAS" --arg mu "$MINIO_URL" \
    --arg mak "$MINIO_ACCESS_KEY" --arg msk "$MINIO_SECRET_KEY" \
    --arg sa "$S3_ALIAS" --arg su "$S3_URL" \
    --arg sak "$S3_ACCESS_KEY" --arg ssk "$S3_SECRET_KEY" '
    ($base | .version = "10" | .aliases //= {})
    | .aliases[$ma] = {
        url: $mu, accessKey: $mak, secretKey: $msk,
        api: "S3v4", path: "auto"
      }
    | if ($sak | length) > 0 then
        .aliases[$sa] = {
          url: $su, accessKey: $sak, secretKey: $ssk,
          api: "S3v4", path: "auto"
        }
      else . end
  '
)" || die "config.json 생성 실패 (jq)"

printf '%s\n' "$NEW_JSON" > "$CONFIG_FILE"
chmod 600 "$CONFIG_FILE"
log "설정 저장: $CONFIG_FILE (600)"

# ── 5. 검증 ─────────────────────────────────────────────────────
log "등록된 alias:"
"$MC" alias list | sed 's/^/    /'

log "연결 테스트: $MINIO_ALIAS"
if "$MC" ls "$MINIO_ALIAS" >/dev/null 2>&1; then
  log "MinIO 연결 정상"
else
  warn "MinIO 연결 실패 — URL/키/방화벽을 확인하세요."
  warn "자체서명 인증서라면: $MC --insecure ls $MINIO_ALIAS"
fi

if [[ -n "$S3_ACCESS_KEY" ]]; then
  log "연결 테스트: $S3_ALIAS"
  "$MC" ls "$S3_ALIAS" >/dev/null 2>&1 \
    && log "S3 연결 정상" \
    || warn "S3 연결 실패 — 키/리전/권한을 확인하세요."
fi

echo
log "완료. 사용 예: $MC ls $MINIO_ALIAS"

```

```
# Write ParallelCluster Config File

cat << 'EOF' > ~/cluster-config.yaml
Region: <아까 나온 리전, 예: ap-northeast-2>
Image:
  Os: alinux2
HeadNode:
  InstanceType: t3.medium
  Networking:
    SubnetId: <SUBNET_ID>
  Ssh:
    KeyName: SBI_KLS_KEY
Scheduling:
  Scheduler: slurm
  SlurmQueues:
    - Name: ess-queue
      ComputeResources:
        - Name: ess-compute
          InstanceType: c5.9xlarge
          MinCount: 0
          MaxCount: 30
      Networking:
        SubnetIds:
          - <SUBNET_ID>
EOF

```

```
# Region check
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region
```

```
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].KeyName' --output text
# KeyPair Check

#if None then create one
aws ec2 create-key-pair --key-name ess-cluster-key \
  --query 'KeyMaterial' --output text > ~/ess-cluster-key.pem
chmod 400 ~/ess-cluster-key.pem
```

```
TOKEN=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
MAC=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/mac)
VPC_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/network/interfaces/macs/$MAC/vpc-id)
SUBNET_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/network/interfaces/macs/$MAC/subnet-id)

echo "VPC: $VPC_ID"
echo "Subnet: $SUBNET_ID"

# AWS ParallelCluster Setup - Information Extractor
```

```
#!/bin/bash
set -euo pipefail

SRC_BASE="s3/path"
DST_BASE="s3/new_path"
STATE_FILE="$HOME/.mc_sync_state"
LOCK_FILE="$HOME/.mc_sync.lock"
PARALLEL=20
DEFAULT_START_DATE="20240101"   # 최초 실행 시 시작점

# 동시 실행 방지 (크론이 겹쳐 돌아가는 것 방지)
exec 200>"$LOCK_FILE"
flock -n 200 || { echo "이미 실행 중입니다. 종료."; exit 1; }

TODAY=$(date +%Y%m%d)

if [ -f "$STATE_FILE" ]; then
  START_DATE=$(cat "$STATE_FILE")
else
  START_DATE="$DEFAULT_START_DATE"
fi

echo "[$(date)] 동기화 시작: ${START_DATE} ~ ${TODAY}"

d="$START_DATE"
while [ "$d" -le "$TODAY" ]; do
  echo "  -> 날짜 처리 중: $d"

  # 해당 날짜 폴더의 sitecode 목록만 조회 (비재귀, 가벼움)
  SITECODES=$(mc ls "${SRC_BASE}/${d}/" 2>/dev/null | awk '{print $NF}' | sed 's:/$::')

  if [ -z "$SITECODES" ]; then
    echo "     (해당 날짜 데이터 없음, 스킵)"
    d=$(date -d "${d} +1 day" +%Y%m%d)
    continue
  fi

  export SRC_BASE DST_BASE d
  echo "$SITECODES" | xargs -P "$PARALLEL" -I{} bash -c '
    sitecode="$1"
    src="${SRC_BASE}/${d}/${sitecode}/000/bms/${sitecode}_bms_rs_${d}.parquet"
    dst="${DST_BASE}/${sitecode}/${sitecode}_bms_rs_${d}.parquet"
    mc cp "$src" "$dst" >/dev/null 2>&1 \
      && echo "     [OK] $sitecode" \
      || echo "     [스킵/실패] $sitecode (원본 없거나 아직 미도착)"
  ' _ {}

  d=$(date -d "${d} +1 day" +%Y%m%d)
done

# 오늘 하루는 아직 데이터가 덜 들어왔을 수 있으니 확정하지 않고,
# "오늘"부터 다시 검증하도록 상태 저장 (오늘자는 다음 실행에 재확인)
echo "$TODAY" > "$STATE_FILE"

echo "[$(date)] 동기화 완료. 다음 실행 시 ${TODAY}부터 재검증."

```


```
import os

def main():
    # 시스템 전체 코어 수 (taskset 영향 없음)
    total_cores = os.cpu_count()

    # 실제로 이 프로세스가 사용 가능한 코어 집합 (taskset 영향 받음)
    available_cores = os.sched_getaffinity(0)
    usable_count = len(available_cores)

    print(f"시스템 전체 코어 수 (os.cpu_count()): {total_cores}")
    print(f"실제 사용 가능한 코어 번호 (os.sched_getaffinity(0)): {sorted(available_cores)}")
    print(f"실제 사용 가능한 코어 개수: {usable_count}")

if __name__ == "__main__":
    main()

```


```
import numpy as np

def _step_score_from_series(x, values, edge_frac=0.2, eps=1e-8):
    """head/tail 구간을 합쳐 공통 슬로프+레벨차를 추정하고,
    diff 기반 scale-free jump 탐지와 결합해 step_score를 계산하는 공용 헬퍼."""
    n = len(values)
    edge = max(int(n * edge_frac), 5)

    x_head, v_head = x[:edge], values[:edge]
    x_tail, v_tail = x[-edge:], values[-edge:]

    X = np.concatenate([x_head, x_tail])
    V = np.concatenate([v_head, v_tail])
    is_tail = np.concatenate([np.zeros(edge), np.ones(edge)])

    A = np.column_stack([X, np.ones_like(X), is_tail])
    coeffs, *_ = np.linalg.lstsq(A, V, rcond=None)
    resid = V - A @ coeffs
    dof = max(len(V) - 3, 1)
    sigma_resid = max(np.sqrt(np.sum(resid**2) / dof), eps)
    persistence_ratio = np.abs(coeffs[2]) / sigma_resid

    dv = np.diff(values)
    med_dv = np.median(dv)
    mad_dv = max(np.median(np.abs(dv - med_dv)) * 1.4826, eps)
    jump_z = np.max(np.abs(dv - med_dv)) / mad_dv

    return min(jump_z, persistence_ratio)


def calc_fit_metrics(x, d, rul, eps=1e-8, edge_frac=0.2):
    err = rul - d
    n = len(err)

    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    max_err = np.max(np.abs(err))

    if np.std(d) < eps or np.std(rul) < eps:
        corr = np.nan
    else:
        corr = np.corrcoef(d, rul)[0, 1]

    ss_res = np.sum((d - rul) ** 2)
    ss_tot = np.sum((d - np.mean(d)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > eps else np.nan

    step_score_model = _step_score_from_series(x, err, edge_frac, eps)
    step_score_data = _step_score_from_series(x, d, edge_frac, eps)

    return {
        "rmse": rmse, "mae": mae, "bias": bias, "max_error": max_err,
        "corr": corr, "r2": r2,
        "step_score_model": step_score_model,
        "step_score_data": step_score_data,
    }

```

```
def _robust_step_score(err, edge_frac=0.2, eps=1e-8):
    """잔차 배열에서 CUSUM + persistence 기반 step_score를 계산하는 공용 헬퍼."""
    n = len(err)
    err_diff = np.diff(err)
    mad = np.median(np.abs(err_diff - np.median(err_diff)))
    sigma_robust = max(mad * 1.4826 / np.sqrt(2), np.std(err) * 0.05, eps)

    cusum = np.cumsum(err - np.mean(err))
    cusum_stat = np.max(np.abs(cusum)) / (sigma_robust * np.sqrt(n))

    edge = max(int(n * edge_frac), 3)
    head = np.median(err[:edge])
    tail = np.median(err[-edge:])
    persistence_ratio = np.abs(tail - head) / sigma_robust

    return min(cusum_stat, persistence_ratio)


def calc_fit_metrics(x, d, rul, eps=1e-8, edge_frac=0.2):
    err = rul - d
    n = len(err)

    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    max_err = np.max(np.abs(err))

    if np.std(d) < eps or np.std(rul) < eps:
        corr = np.nan
    else:
        corr = np.corrcoef(d, rul)[0, 1]

    ss_res = np.sum((d - rul) ** 2)
    ss_tot = np.sum((d - np.mean(d)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > eps else np.nan

    step_score_model = _robust_step_score(err, edge_frac, eps)

    slope, intercept, _, _ = theilslopes(d, x)
    err_raw = d - (slope * x + intercept)
    step_score_data = _robust_step_score(err_raw, edge_frac, eps)

    return {
        "rmse": rmse, "mae": mae, "bias": bias, "max_error": max_err,
        "corr": corr, "r2": r2,
        "step_score_model": step_score_model,
        "step_score_data": step_score_data,
    }


```

```
def calc_fit_metrics(x, d, rul, eps=1e-8, edge_frac=0.2):
    err = rul - d
    n = len(err)

    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    max_err = np.max(np.abs(err))

    if np.std(d) < eps or np.std(rul) < eps:
        corr = np.nan
    else:
        corr = np.corrcoef(d, rul)[0, 1]

    ss_res = np.sum((d - rul) ** 2)
    ss_tot = np.sum((d - np.mean(d)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > eps else np.nan

    err_diff = np.diff(err)
    mad = np.median(np.abs(err_diff - np.median(err_diff)))
    sigma_robust = max(mad * 1.4826 / np.sqrt(2), np.std(err) * 0.05, eps)

    cusum = np.cumsum(err - np.mean(err))
    cusum_stat = np.max(np.abs(cusum)) / (sigma_robust * np.sqrt(n))

    edge = max(int(n * edge_frac), 3)
    head_level = np.median(err[:edge])
    tail_level = np.median(err[-edge:])
    persistence_ratio = np.abs(tail_level - head_level) / sigma_robust

    step_score = min(cusum_stat, persistence_ratio)

    return {
        "rmse": rmse, "mae": mae, "bias": bias, "max_error": max_err,
        "corr": corr, "r2": r2,
        "step_score": step_score,
    }

```


```
import numpy as np
import pandas as pd


def get_iqr_bounds(s):
    s = pd.to_numeric(s, errors="coerce").dropna()

    q1 = s.quantile(0.25)
    q2 = s.quantile(0.50)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return q1, q2, q3, iqr, lower, upper


def extract_metric_tolerances(df):
    upper_metrics = [
        "rmse",
        "mae",
        "max_error",
        "grad_std",
        "curvature_noise",
        "diff_std",
    ]

    lower_metrics = [
        "corr",
        "r2",
    ]

    two_side_metrics = [
        "bias",
    ]

    rows = []

    for col in upper_metrics:
        q1, q2, q3, iqr, lower, upper = get_iqr_bounds(df[col])

        rows.append({
            "metric": col,
            "direction": "upper",
            "tolerance_lower": np.nan,
            "tolerance_upper": upper,
            "q1": q1,
            "median": q2,
            "q3": q3,
            "iqr": iqr,
            "p05": df[col].quantile(0.05),
            "p95": df[col].quantile(0.95),
        })

    for col in lower_metrics:
        q1, q2, q3, iqr, lower, upper = get_iqr_bounds(df[col])

        rows.append({
            "metric": col,
            "direction": "lower",
            "tolerance_lower": lower,
            "tolerance_upper": np.nan,
            "q1": q1,
            "median": q2,
            "q3": q3,
            "iqr": iqr,
            "p05": df[col].quantile(0.05),
            "p95": df[col].quantile(0.95),
        })

    for col in two_side_metrics:
        q1, q2, q3, iqr, lower, upper = get_iqr_bounds(df[col])

        rows.append({
            "metric": col,
            "direction": "two_side",
            "tolerance_lower": lower,
            "tolerance_upper": upper,
            "q1": q1,
            "median": q2,
            "q3": q3,
            "iqr": iqr,
            "p05": df[col].quantile(0.05),
            "p95": df[col].quantile(0.95),
        })

    tol_df = pd.DataFrame(rows)

    return tol_df



tol_df = extract_metric_tolerances(df)
print(tol_df)
tol_df.to_csv("metric_tolerances.csv", index=False)





def check_metric_tolerance(row, tol_df):
    violations = []

    for _, t in tol_df.iterrows():
        metric = t["metric"]
        direction = t["direction"]
        value = row[metric]

        if direction == "upper":
            if value > t["tolerance_upper"]:
                violations.append(metric)

        elif direction == "lower":
            if value < t["tolerance_lower"]:
                violations.append(metric)

        elif direction == "two_side":
            if value < t["tolerance_lower"] or value > t["tolerance_upper"]:
                violations.append(metric)

    return violations



df["violations"] = df.apply(
    lambda row: check_metric_tolerance(row, tol_df),
    axis=1
)

df["n_violations"] = df["violations"].apply(len)
```


```
df_sorted = df.sort_values(
    by="n_violations",
    ascending=False
).reset_index(drop=True)


import pandas as pd
import matplotlib.pyplot as plt


def plot_file_curve(file_path, title=None):
    fdf = pd.read_csv(file_path)

    plt.figure(figsize=(8, 5))

    plt.plot(
        fdf["fieldf_x"],
        fdf["fieldf_y"],
        marker="o",
        linestyle="-",
        label="data"
    )

    plt.plot(
        fdf["cmodel_x"],
        fdf["cmodel_y"],
        linestyle="--",
        label="rul"
    )

    if title is None:
        title = file_path

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



for _, row in df_sorted.head(10).iterrows():
    file_path = row["file"]
    n_v = row["n_violations"]

    title = f"{file_path} | n_violations={n_v}"

    plot_file_curve(file_path, title=title)



target_n = 3

df_target = df[df["n_violations"] == target_n]

for _, row in df_target.iterrows():
    file_path = row["file"]

    title = f"{file_path} | n_violations={target_n}"

    plot_file_curve(file_path, title=title)




def plot_by_violation_group(df, max_per_group=5):
    df_sorted = df.sort_values(
        by="n_violations",
        ascending=False
    )

    for n_v, g in df_sorted.groupby("n_violations", sort=False):
        print(f"\n===== n_violations = {n_v} =====")

        for _, row in g.head(max_per_group).iterrows():
            file_path = row["file"]

            title = f"{file_path} | n_violations={n_v}"

            plot_file_curve(file_path, title=title)

plot_by_violation_group(df, max_per_group=3)





def plot_from_summary_row(row):
    file_path = row["file"]

    title = (
        f"{file_path}\n"
        f"n_violations={row['n_violations']} | "
        f"rmse={row['rmse']:.4g}, "
        f"mae={row['mae']:.4g}, "
        f"corr={row['corr']:.4g}, "
        f"r2={row['r2']:.4g}"
    )

    plot_file_curve(file_path, title=title)

for _, row in df_sorted.head(10).iterrows():
    plot_from_summary_row(row)
```


```
def add_weighted_violation_score(df, weights, violations_col="violations"):
    """
    df["violations"]에 들어있는 metric 리스트를 기준으로
    metric별 weight를 합산하여 score_violations 컬럼 추가.
    """

    def calc_score(vs):
        if isinstance(vs, str):
            # CSV에서 읽어서 "['rmse', 'mae']" 같은 문자열인 경우 처리
            import ast
            try:
                vs = ast.literal_eval(vs)
            except Exception:
                vs = [vs]

        if vs is None:
            return 0.0

        return sum(weights.get(v, 1.0) for v in vs)

    df = df.copy()
    df["score_violations"] = df[violations_col].apply(calc_score)

    return df


weights = {
    "rmse": 3.0,
    "mae": 2.0,
    "bias": 2.0,
    "max_error": 2.5,
    "corr": 1.5,
    "r2": 1.5,
    "grad_std": 1.0,
    "curvature_noise": 3.0,
    "diff_std": 2.0,
}

df = add_weighted_violation_score(df, weights)

df_sorted = df.sort_values(
    by="score_violations",
    ascending=False
).reset_index(drop=True)

df_sorted[
    ["file", "n_violations", "score_violations", "violations"]
].head(20)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(df["score_violations"], bins=20)
plt.xlabel("score_violations")
plt.ylabel("count")
plt.title("Weighted Violation Score Distribution")
plt.grid(True)
plt.show()
```



```
import ast
import pandas as pd

def add_weighted_violation_score(
    df,
    weights,
    violations_col="violations",
    grad_std_threshold=5.0,
):
    """
    df["violations"]에 들어있는 metric 리스트를 기준으로
    metric별 weight를 합산하여 score_violations 컬럼 추가.

    단, grad_std는 실제 row["grad_std"] 값이 grad_std_threshold보다 클 때만
    weight를 반영.
    """

    def parse_violations(vs):
        if isinstance(vs, str):
            try:
                vs = ast.literal_eval(vs)
            except Exception:
                vs = [vs]

        if vs is None:
            return []

        if isinstance(vs, float) and pd.isna(vs):
            return []

        return vs

    def calc_score(row):
        vs = parse_violations(row[violations_col])

        score = 0.0

        for v in vs:
            if v == "grad_std":
                if row["grad_std"] > grad_std_threshold:
                    score += weights.get(v, 1.0)
            else:
                score += weights.get(v, 1.0)

        return score

    df = df.copy()

    df["score_violations"] = df.apply(
        calc_score,
        axis=1
    )

    return df

weights = {
    "rmse": 3.0,
    "mae": 2.0,
    "bias": 2.0,
    "max_error": 2.5,
    "corr": 1.5,
    "r2": 1.5,
    "grad_std": 1.0,
    "curvature_noise": 3.0,
    "diff_std": 2.0,
}

df = add_weighted_violation_score(
    df,
    weights,
    grad_std_threshold=5.0
)
```

