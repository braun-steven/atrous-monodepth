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
declare -a arr=("06" "07" "08" "09" "10" "20")

# First argument is your username
BOLD="\e[1m"
NC="\e[0m"
GREEN="\e[92m"
BLUE="\e[94m"
RED="\e[91m"
DEFAULTCOLOR="\e[39m"

# Loop through each server
for i in "${arr[@]}"
do
  server="${username}@lab${i}.visinf.informatik.tu-darmstadt.de" 
  echo -e "\n"
  echo -e "${BOLD}## (${BLUE}lab${i}${DEFAULTCOLOR}) Current status: ${NC}\n"
  ssh -o ConnectTimeout=5 $server nvidia-smi | grep MiB | grep % | cut -d'|' -f3
done

echo -e "\n\n"
echo -e "${BOLD}${GREEN}Finished scanning all servers ...${NC}"
