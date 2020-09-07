#!/bin/bash

export THISDIR=`pwd`
export CORE=9-15
export TIME_LIMIT=300
export MEMORY_LIMIT=32000
export MEMLIMIT=32000
export DELAY=60
export RESULTDIR=${HOME}/cp2020/results_fixeddiffprobshesha_generated_4_1024_300
export RUNLOG=${RESULTDIR}/runall.log
export BENCHMARK_DIR=${HOME}/cp2020/benchmarks/generated/
export RUNSOLVER_EXEC=${HOME}/bin/runsolver
export TMPCNF=/tmp/maxsat.cnf
mkdir -p ${RESULTDIR}

export shesha_alpha="$1"
export shesha_beta="$2"
export shesha_sp="$3"
export shesha_hard_prob="$4"
export shesha_hard_inc="$5"
export shesha_nblocks="$6"
export shesha_nthreads="$7"
export shesha_n="$8"
export shesha_soft_max_delta="$9"

export TMPWATCH=/tmp/__${shesha_alpha}_${shesha_beta}_${shesha_sp}_${shesha_hard_prob}_${shesha_hard_inc}_${shesha_nblocks}_${shesha_nthreads}_${shesha_n}_${shesha_soft_max_delta}_solver_tmp.watch
export TMPTIME=/tmp/__${shesha_alpha}_${shesha_beta}_${shesha_sp}_${shesha_hard_prob}_${shesha_hard_inc}_${shesha_nblocks}_${shesha_nthreads}_${shesha_n}_${shesha_soft_max_delta}_solver_tmp.time
export TMPOUT=/tmp/__${shesha_alpha}_${shesha_beta}_${shesha_sp}_${shesha_hard_prob}_${shesha_hard_inc}_${shesha_nblocks}_${shesha_nthreads}_${shesha_n}_${shesha_soft_max_delta}_solver_tmp.out

# For Shesha
export TOOL=shesha
export TOOLBINARY=${HOME}/cp2020/tools/shesha/shesha
export TOOLCOMMAND="${TOOLBINARY} --alpha ${shesha_alpha} --beta ${shesha_beta} --sp ${shesha_sp} --hard_prob ${shesha_hard_prob} --hard_inc ${shesha_hard_inc} --nblocks ${shesha_nblocks} --nthreads ${shesha_nthreads} -n ${shesha_n} --soft_max_delta ${shesha_soft_max_delta}"
export RESULTDIR_PREFIX="result_shesha_${shesha_alpha}_${shesha_beta}_${shesha_sp}_${shesha_hard_prob}_${shesha_hard_inc}_${shesha_nblocks}_${shesha_nthreads}_${shesha_n}_${shesha_soft_max_delta}_`date '+%Y-%m-%d'`"

echo "Starting ${TOOL} with ${TOOLCOMMAND} at `date`" 2>&1 >> ${RUNLOG}
bash run_bench_tool.sh 
cd ${THISDIR}
echo "Finished ${TOOL} with ${TOOLCOMMAND} at `date`" 2>&1 >> ${RUNLOG}
