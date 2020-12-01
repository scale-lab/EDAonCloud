# Characterization

In this part, we characterize the performance of the different stages of an EDA flow. In order to reproduce, edit the script file and fill in the command name for the tool of your choice.

## How to run

Execute the scripts on a Linux system where `perf` tool is installed. For example:

```Shell
./1_synth.sh test-run 4 0-3
```
where `test-run` is an arbitrary ID (used to uniquely identify this run), `4` represents the number of CPUs and `0-3` represents the CPU IDs to pin this process on. CPU pinning is important because otherwise, the operating system will keep floating the process between cores depending on the least busy core.


## Limiting Resources with Control Groups 

In order to simulate a cloud environment where resources are virtualized and constrained per user, we utilize [Linux control groups](https://wiki.archlinux.org/index.php/cgroups).

1. Create a cgroup using: `cgcreate -t uid:gid -a uid:gid -g subsystems:path`. Read more about this command in [this link](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/resource_management_guide/sec-creating_cgroups).
2. Edit cgroup resources. For example, edit limits files in `/sys/fs/cgroup/cpu/groupname`.
3. Use `cgexec` to run the experiment. For example, `cgexec -g memory,cpu:groupname 1_synth.sh`

## Output

The output of the performance counters is reported in a file named `$RUN_ID-$STAGE-cpu-$CPU_COUNT.perf.data`.
We provide data for the Sparc Core design in the [data](./data) folder.
Use your favorite scripting language to process and aggregate the data.

For example, to parse the file into a `.csv` format, you might use: `awk -F ' ' '{ print $2", " $1 }' synth-cpu-1.perf.data | tail -n+3 > data.csv`.

## Results

Please, refer to the paper cited in the [README](../README.md) for our crunched results. 

