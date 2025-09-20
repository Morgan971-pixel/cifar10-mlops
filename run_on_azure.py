from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential
import os # Import os for path manipulation

# --- Connection Details ---
# These should ideally come from environment variables in a real CI/CD pipeline
subscription_id = "2b850e23-397c-4461-a543-00a8bf14ee7b"
resource_group = "MyFirstScript"
workspace = "MyFirstScript"
cluster_name = "cpu-cluster"

# --- Authenticate and Connect ---
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace,
)

# --- Verify Connection and Create Compute Cluster (if not exists) ---
try:
    workspace_details = ml_client.workspaces.get(workspace)
    print(f"Successfully connected to workspace: '{workspace_details.name}'")
    print(f"Location: {workspace_details.location}")
except Exception as ex:
    print(f"Failed to connect to workspace. Please check your details. Error: {ex}")
    exit()

try:
    cluster = ml_client.compute.get(cluster_name)
    print(f"\nFound existing compute cluster '{cluster_name}', using it.")
except Exception:
    print(f"\nCreating a new compute cluster '{cluster_name}'...")
    compute_config = AmlCompute(
        name=cluster_name,
        size="Standard_DS3_v2",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=120,
    )
    ml_client.compute.begin_create_or_update(compute_config).result()
    print("Compute cluster created successfully.")

# --- Define Environment ---
# Construct absolute path to requirements.txt relative to the script's location
# This ensures it works correctly in GitHub Actions runner
script_dir = os.path.dirname(os.path.abspath(__file__))
requirements_path = os.path.join(script_dir, "cifar10_cnn_project", "requirements.txt")

# Read requirements.txt content
with open(requirements_path, "r") as f:
    requirements_content = f.read()

# Create a conda.yaml content dynamically
conda_yaml_content = f"""
name: cifar10-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
"""
for line in requirements_content.splitlines():
    if line.strip(): # Only add non-empty lines
        conda_yaml_content += f"    - {line.strip()}\n"

# Write the conda.yaml content to a temporary file in the current working directory
# This ensures it's written to a path accessible by the Azure ML SDK
conda_file_path = "conda.yaml" # Relative to the current working directory of the script
with open(conda_file_path, "w") as f:
    f.write(conda_yaml_content)

# Create the environment
job_env = Environment(
    name="cifar10-cnn-env",
    description="Environment for CIFAR-10 CNN training with Keras Tuner",
    conda_file=conda_file_path, # Pass the path to the temporary file
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)
ml_client.environments.create_or_update(job_env)

# --- Define Command Job ---
# The 'code' parameter should be a path relative to the repository root
# which is the working directory of the GitHub Actions runner.
job = command(
    code="./cifar10_cnn_project",  # Relative path to the code directory
    command="python train.py",
    environment=f"{job_env.name}@latest",
    compute=cluster_name,
    display_name="cifar10-cnn-transfer-learning",
    experiment_name="cifar10-cnn-experiment",
)

# --- Submit Job ---
print("\nSubmitting the job to Azure ML...")
returned_job = ml_client.jobs.create_or_update(job)
print("Job submitted! Check its status in Azure ML Studio.")