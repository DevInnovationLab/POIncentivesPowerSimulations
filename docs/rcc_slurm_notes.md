# RCC SLURM Notes

## Key sbatch flags
- `--cpus-per-task=N` — cores per task (for multiprocessing/OpenMP)
- `--ntasks=N` — total MPI tasks
- `--ntasks-per-node=N` — MPI tasks per node
- `--mem=XG` — total memory per node
- `--mem-per-cpu=X` — memory per core (MB)
- `--array=0-N` — job arrays; `%A_%a` for output naming
- `--time=HH:MM:SS` — wall time

## Parallel approaches
1. **Job arrays** (`--array`): best for embarrassingly parallel, each task gets own allocation
2. **GNU parallel**: `parallel --delay 0.2 -j $SLURM_NTASKS --joblog runtask.log`
3. **Concurrent processes**: background with `&`, bind with `taskset -c $idx`, `wait`
4. **Dependency chains**: `sbatch --dependency=afterany:JOBID next.sbatch`

## Python multiprocessing pattern
```bash
for idx in {0..31}; do
  taskset -c $idx python script.py input-$idx.txt > output-$idx.txt &
done
wait
```

## Best practices
- `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`
- `ulimit -u 10000` for many tasks
- Use `--joblog` + `--resume` with GNU parallel for recovery
- Array task limits exist per partition
