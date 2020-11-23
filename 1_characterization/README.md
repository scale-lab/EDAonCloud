# Characterization

In this part, we characterize the performance of the different stages of an EDA flow. In order to reproduce, edit the script file and fill in the command name for the tool of your choice.

## How to run

Execute the scripts on a Linux system where `perf` tool is installed. For example:

```Shell
./1_synth.sh
```

## Limiting Resources with Control Groups 

In order to simulate a cloud environment where resources are virtualized and constrained per user, we utilize [Linux control groups](https://wiki.archlinux.org/index.php/cgroups).

1. Create a cgroup using: `cgcreate -t uid:gid -a uid:gid -g subsystems:path`. Read more about this command in [this link](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/resource_management_guide/sec-creating_cgroups).
2. Edit cgroup resources. For example, edit limits files in `/sys/fs/cgroup/cpu/groupname`.
3. Use `cgexec` to run the experiment. For example, `cgexec -g memory,cpu:groupname 1_synth.sh`

## Output

The output of the performance counters is reported in a file named `$STAGE-cpu-$CPU_COUNT.perf.data`.
You may use your preferred scripting language to parse these files.

## Results

Please, refer to the paper cited in the [README](../README.md) for our crunched results.

