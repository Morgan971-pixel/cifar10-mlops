from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

# --- Azure ML Job Submission ---
def main():
    # Connect to Azure ML
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id="2b850e23-397c-4461-a543-00a8bf14ee7b",
        resource_group_name="MyFirstScript",
        workspace_name="MyFirstScript",
    )

    # Define the command to run the training script
    job_command = "python train.py"

    # Create the command job
    job = command(
        command=job_command,
        environment="my-tf-environment:1", 
        compute="cpu-cluster",
        display_name="cifar10-cnn-training-cpu-manual-env",
        description="Train a CNN on CIFAR-10 using a CPU cluster with a manually created environment.",
        experiment_name="cifar10-cnn-experiment",
        code="."
    )

    # Submit the job
    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job submitted with ID: {returned_job.name}")
    print(f"You can view the job here: {returned_job.studio_url}")

if __name__ == "__main__":
    main()