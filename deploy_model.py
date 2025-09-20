from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model, Environment, CodeConfiguration, ManagedOnlineEndpoint, ManagedOnlineDeployment, ProbeSettings
from azure.identity import DefaultAzureCredential
import os
import time

# --- Connection Details ---
subscription_id = "2b850e23-397c-4461-a543-00a8bf14ee7b"
resource_group = "MyFirstScript"
workspace = "MyFirstScript"

# --- Authenticate and Connect ---
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace,
)

# --- Retrieve Registered Model ---
model_name = "cifar10-cnn-model"
# Get the latest version of the model
try:
    registered_model = ml_client.models.get(name=model_name, label="latest")
    print(f"Retrieved model: {registered_model.name}, version: {registered_model.version}")
except Exception as e:
    print(f"Could not retrieve model '{model_name}' with label 'latest'. Please ensure the training job completed successfully and registered the model. Error: {e}")
    exit()

# --- Define Inference Environment ---
# Construct absolute path to inference_requirements.txt relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
inference_requirements_path = os.path.join(script_dir, "cifar10_cnn_project", "inference_requirements.txt")

# Create a conda.yaml content dynamically for inference environment
with open(inference_requirements_path, "r") as f:
    inference_requirements_content = f.read()

inference_conda_yaml_content = f"""
name: inference-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
"""
for line in inference_requirements_content.splitlines():
    if line.strip():
        inference_conda_yaml_content += f"    - {line.strip()}\n"

# Write the conda.yaml content to a temporary file for inference environment
inference_conda_file_path = "inference_conda.yaml" # Relative to the current working directory of the script
with open(inference_conda_file_path, "w") as f:
    f.write(inference_conda_yaml_content)

inference_env = Environment(
    name="cifar10-inference-env",
    description="Environment for CIFAR-10 CNN model inference",
    conda_file=inference_conda_file_path,
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
)
ml_client.environments.create_or_update(inference_env)

# --- Define Online Endpoint ---
# Endpoint names must be unique in the Azure region
# Using a timestamp to ensure uniqueness
endpoint_name = "cifar10-endpoint-" + str(int(time.time()))

endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Online endpoint for CIFAR-10 CNN model",
    auth_mode="key",
)

# --- Create/Update Endpoint ---
print(f"\nCreating/Updating endpoint '{endpoint_name}'...")
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Endpoint created/updated successfully.")

# --- Define Online Deployment ---
deployment_name = "blue"
deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=endpoint_name,
    model=registered_model,
    environment=inference_env,
    code_configuration=CodeConfiguration(
        code="./cifar10_cnn_project", # Path to the directory containing score.py
        scoring_script="score.py",
    ),
    instance_type="Standard_DS3_v2", # Choose an appropriate VM size
    instance_count=1,
    liveness_probe=ProbeSettings(initial_delay=10, period=60, timeout=60, failure_threshold=10, success_threshold=1),
    readiness_probe=ProbeSettings(initial_delay=10, period=60, timeout=60, failure_threshold=10, success_threshold=1),
)

# --- Create/Update Deployment ---
print(f"\nCreating/Updating deployment '{deployment_name}' for endpoint '{endpoint_name}'...")
ml_client.online_deployments.begin_create_or_update(deployment).result()
print("Deployment created/updated successfully.")

# --- Set Traffic to New Deployment ---
print(f"\nSetting traffic to deployment '{deployment_name}'...")
endpoint.traffic = {deployment_name: 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("Traffic set successfully.")

print(f"\nDeployment complete! Endpoint URL: {endpoint.scoring_uri}")