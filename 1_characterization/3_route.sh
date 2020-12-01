#!/bin/sh
if [ "$#" -lt 3 ]; then
    echo "Usage: ./run.sh RUN_ID CPU_COUT CPU_LIST" >&2
    exit 1
fi

set -x
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT


DESIGN=${PWD##*/}
RUN_ID=$1
CPU_COUNT=$2
CPUS=$3

# Routing script (depends on the tool used)
TCL_FILE=3_route.tcl

# Performance Counters
PC="instructions,cpu-cycles,ref-cycles,bus-cycles,cache-references,cache-misses,branches,branch-misses"
PC=$PC",L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores"
PC=$PC",L1-icache-load-misses"
PC=$PC",LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses"
PC=$PC",dTLB-loads,dTLB-load-misses,dTLB-stores,dTLB-store-misses,iTLB-loads,iTLB-load-misses"
PC=$PC",branch-loads,branch-load-misses"
PC=$PC",block:block_rq_issue,block:block_rq_complete"
PC=$PC",fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.scalar_single"
PC=$PC",fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.128b_packed_single"
PC=$PC",fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.256b_packed_single"


# routing
run_command=''  # put tool command here
perf stat -r 3 -o $RUN_ID-route-$CPU_COUNT.perf.data -e $PC -C $CPUS taskset -c $CPUS $run_command
