# Optimization

In order to run the optimizer, provide a data file similar to `example_data.csv` file. 
The first row contains the classes of the jobs (e.g. synthesis, placement, routing, sta).
They are encoded starting from 0.
The second row represents the runtime it took on a specific machine configuration (e.g. 1, 2, 4, 8 vCPUs). Note that the order matters as the optimization uses the inices of these runtimes to relate them to a machine congiguration.
The third row represents the hourly cost for running that configuration for that stage.

To run:

```shell
python cloud_optimizer.py <budget> -data <data_file>
```

For example:

```
python cloud_optimizer.py 3000 -data example_data.csv
```

This will output the total runtime, the total cost, and the recommended machine configuration (index) for each class.

