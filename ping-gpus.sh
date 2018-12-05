#!/bin/bash

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

function show_help {
	echo "Usage: ping-gpus.sh -u username"
}

# Initialize our own variables:
username=""

while getopts "h?u:" opt; do
  case "$opt" in
	h|\?)
	  show_help
	  exit 0
	  ;;
	u)  username=$OPTARG
	  ;;
  esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--"  ] && shift

if [[ "$username" = "" ]]
then
	echo $username
	echo "Username not set."
	show_help
	exit 1
fi

# Server idx
declare -a arr=("01" "03" "04" "05" "06" "07" "08" "09" "10" "20")

# First argument is your username
BOLD="\e[1m"
NC="\e[0m"
GREEN="\e[92m"
BLUE="\e[94m"
RED="\e[91m"
DEFAULTCOLOR="\e[39m"
GRAY="\e[37m"

function printmeminfo {
  # Get nvidia-smi output
  result=$(ssh -o ConnectTimeout=5 ${username}@lab$1.visinf.informatik.tu-darmstadt.de nvidia-smi)
  # Get GPU indices
  idxs=( $(echo "$result" | grep GeForce | cut -d'G' -f1 | cut -d' ' -f4) )
  # Get GPU memory info
  memory=$(echo "$result" | grep MiB | grep % | cut -d'|' -f3 )
  # Parse GPU memory info
  memarr=()
  while read -r line; do
    memarr+=("$line")
  done <<< "$memory"

  # Print info
  for i in "${!idxs[@]}"
  do
    echo -e "${GRAY}GPU (${idxs[$i]}):${NC} ${memarr[$i]}"
  done
}


# Loop through each server
for i in "${arr[@]}"
do
  echo -e "\n"
  echo -e "${BOLD}## (${BLUE}lab${i}${DEFAULTCOLOR}) Current status: ${NC}\n"
  printmeminfo $i
done

echo -e "\n\n"
echo -e "${BOLD}${GREEN}Finished scanning all servers ...${NC}"
