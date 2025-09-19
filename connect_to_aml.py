from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute

# --- Connection Details ---
# You can find these details in the Azure Portal on the overview page of your Azure Machine Learning workspace
subscription_id = "2b850e23-397c-4461-a543-00a8bf14ee7b"
resource_group = "MyFirstScript" # The resource group your workspace is in
workspace = "MyFirstScript" # The name of your Azure ML workspace

# --- Authenticate and Connect ---
# Uses your credentials from 'az login'
credential = DefaultAzureCredential()

# MLClient is the main entry point to Azure ML
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace,
)

# --- Verify Connection ---
# Print details of the workspace to confirm connection
try:
    workspace_details = ml_client.workspaces.get(workspace)
    print(f"Successfully connected to workspace: '{workspace_details.name}'")
    print(f"Location: {workspace_details.location}")
    print(f"Description: {workspace_details.description}")
except Exception as ex:
    print(f"Failed to connect to workspace. Please check your details. Error: {ex}")


# --- Create a Compute Cluster ---
# A compute cluster is a managed resource for running training jobs.
# It will autoscale from min_nodes to max_nodes.
cluster_name = "cpu-cluster"

try:
    # Check if the cluster already exists
    cluster = ml_client.compute.get(cluster_name)
    print(f"\nFound existing compute cluster '{cluster_name}', using it.")
except Exception:
    # If not, create it
    print(f"\nCreating a new compute cluster '{cluster_name}'...")
    compute_config = AmlCompute(
        name=cluster_name,
        size="Standard_DS3_v2", # A general-purpose VM size
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=120, # Scale down after 2 minutes of inactivity
    )
    ml_client.compute.begin_create_or_update(compute_config).result()
    print("Compute cluster created successfully.")