# GEqTrain Training Commands  

## SBATCH Script Explanation  
Below is a breakdown of the SBATCH parameters used in the script:  
- `--job-name=geqtrain-train`: Sets the name of the job to `geqtrain-train`.  
- `--time=00:05:00`: Specifies the maximum runtime for the job (5 minutes).  
- `--nodes=1`: Requests 1 compute node.  
- `--ntasks-per-node=1`: Specifies 1 task per node.  
- `--cpus-per-task=128`: Allocates 128 CPUs for the task.  
- `--environment=geqtrain`: Loads the `geqtrain` environment.  
- `--account=u8`: Specifies the account to charge for the job.  
- `--partition=debug`: Submits the job to the `debug` partition.  
- `--mem=128GB`: Allocates 128GB of memory for the job.  

## Running the Training Job  
To submit the training job, use the following command:  
```bash  
sbatch geqtrain.sbatch path/to/config.yaml  
```  

## Checking Job Status  
After submitting the job, you can check its status using the following commands:  

1. **List all jobs for your user**:  
    ```bash  
    squeue -u $USER  
    ```  

2. **Filter jobs by job name**:  
    ```bash  
    squeue -u $USER -n geqtrain-train  
    ```  

3. **View detailed job information**:  
    ```bash  
    scontrol show job <job_id>  
    ```  

4. **Check job output and error logs**:  
    After the job completes, check the output and error files generated in the submission directory. For example:  
    ```bash  
    cat slurm-<job_id>.out
    cat slurm-<job_id>.err

5. **Canceling a Job**: 
    To cancel a running job, use the following command with your job ID:  
    ```bash
    scancel <job_id>
    ```