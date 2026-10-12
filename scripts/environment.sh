# Sourced by the scripts that need something of the machine they run on
# (bash): an install prefix, a program, a directory to write to, a device.
#
# A script names no path of one machine (#289, docs/environment.md,
# "Scripts"). What a machine provides comes from the environment:
#
#  - a program is a variable that holds its name or its path and defaults to
#    the bare name, found on the path (need_program);
#  - a directory, or a choice of the machine, is a variable without a
#    default (need_variable, need_directory).
#
# A script states all its needs at its start and then calls need_check,
# which prints every one that is missing, with one line on what it is, and
# exits with status 2, before the script has fetched, removed, or written
# anything.
#
# A machine may keep its values in one file outside the repository: if
# MDIR_ENV_FILE is set, the lines NAME=value of that file are read here.
# The value is the rest of the line as it is (no quotes, no expansion;
# nothing is executed); lines that are empty or start with # are skipped.
# A variable that the environment already has keeps its value, so that one
# call can override the file.

if [[ -n ${MDIR_ENV_FILE:-} ]]; then
  [[ -r $MDIR_ENV_FILE ]] || { echo "$0: MDIR_ENV_FILE names $MDIR_ENV_FILE, which cannot be read" >&2; exit 2; }
  mdir_env_line=0
  while IFS= read -r mdir_env_text || [[ -n $mdir_env_text ]]; do
    mdir_env_line=$((mdir_env_line + 1))
    [[ $mdir_env_text =~ ^[[:space:]]*(#|$) ]] && continue
    if [[ $mdir_env_text =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
      [[ -n ${!BASH_REMATCH[1]+set} ]] || export "${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
    else
      echo "$0: $MDIR_ENV_FILE:$mdir_env_line: not NAME=value: $mdir_env_text" >&2; exit 2
    fi
  done < "$MDIR_ENV_FILE"
  unset mdir_env_line mdir_env_text
fi

mdir_env_missing=()

# need_variable NAME WHAT: NAME must be set and not empty.
need_variable() {
  [[ -n ${!1:-} ]] || mdir_env_missing+=("$1 is not set: $2")
}

# need_directory NAME WHAT: NAME must name a directory that exists.
need_directory() {
  if [[ -z ${!1:-} ]]; then
    mdir_env_missing+=("$1 is not set: $2")
  elif [[ ! -d ${!1} ]]; then
    mdir_env_missing+=("$1 names ${!1}, which is not a directory: $2")
  fi
}

# need_program NAME DEFAULT WHAT: NAME, or DEFAULT when NAME is not set,
# must be a program, by its path or on the path. NAME is then set to the
# absolute path of what was found.
need_program() {
  local program=${!1:-$2} found
  if found=$(command -v "$program" 2> /dev/null) && [[ -x $found ]]; then
    [[ $found != */* || $found == /* ]] || found=$PWD/$found
    printf -v "$1" %s "$found"
  elif [[ -n ${!1:-} ]]; then
    mdir_env_missing+=("$1 names $program, which is not a program: $3")
  else
    mdir_env_missing+=("$1 is not set and $2 is not on the path: $3")
  fi
}

# need_check: stops the script if anything is missing.
need_check() {
  (( ${#mdir_env_missing[@]} )) || return 0
  echo "$0: the environment lacks what this script needs (docs/environment.md, Scripts):" >&2
  printf '  %s\n' "${mdir_env_missing[@]}" >&2
  exit 2
}
