# Train and deploy: The MLOps starter pack

Run **one Azure ML job** in **your subscription** (via GitHub Actions):
- **Train -> Score** the Titanic dataset in a single run.
- Logs metrics to **Azure ML Studio** and saves `preds.csv`.

---

## Repo must include
- `data/Titanic-Dataset.csv`  ← **exact name/case**
- `aml/train_and_score.yml`
- `src/train.py`, `src/score.py`
- `.github/workflows/train-and-score.yml`
- `.github/workflows/nuke-aml.yml`

---

## Prereqs
- An **Azure subscription**
- A **GitHub account**

---

## Create the GitHub secret

Create a Service Principal and paste its JSON into a GitHub **Actions secret** named **`AZURE_CREDENTIALS`**.

**In Azure Cloud Shell (Bash) or any shell with Azure CLI:**
```bash
# Confirm the subscription
az account show -o table

# Register Azure ML resource provider (one-time, may take a few minutes)
az provider register --namespace Microsoft.MachineLearningServices

# Optional: Alerts/Insights to avoid noisy errors
az provider register --namespace Microsoft.AlertsManagement
az provider register --namespace microsoft.insights

# Capture subscription id
SUB_ID=$(az account show --query id -o tsv)

# Create a Contributor SP scoped to your subscription
# --sdk-auth outputs JSON for azure/login@v2
az ad sp create-for-rbac \
  --name "gh-aml-$(whoami)-$(date +%s)" \
  --role Contributor \
  --scopes "/subscriptions/$SUB_ID" \
  --sdk-auth \
  | tee /tmp/azure-credentials.json

# Add the secret in GitHub (UI)
# Repo → Settings → Secrets and variables → Actions → New repository secret
# Name: AZURE_CREDENTIALS
# Value: (paste the JSON printed above)
## Optional (GitHub CLI):
gh secret set AZURE_CREDENTIALS < /tmp/azure-credentials.json


## Run (GitHub Actions)
- GitHub -> Actions -> AML Train_and_Score -> Run workflow.
- Inputs:
-- alias (e.g., p07) -> creates RG rg-ml-p07 and Workspace mlw-p07 in polandcentral.
-- compute_policy: warm (keeps 1 node hot), cold (scale to 0).
-- recommended sizes: Standard_F2s_V2, Standard_D2s_V3, Standard_D2_V3, Standard_DS2_V2.
- When it finishes, download artifact scored_<alias>.

## See results in Azure ML Studio
- Open Azure ML Studio -> select your Workspace (mlw-<alias>).
- Go to Jobs -> open the latest run:
-- Metrics: accuracy, accuracy_head
-- Artifacts: preds_head.csv, metrics.json, sample_request.json
-- Outputs + logs -> scored: full preds.csv (download)


## Clean up
Actions -> AML Nuke RG, enter RG name, type NUKE, Run.

## Optional: Scale down (stop spend, keep workspace)
az ml compute update -n cpu-cluster -g rg-ml-<alias> -w mlw-<alias> \
  --min-instances 0 --max-instances 1 --idle-time-before-scale-down 120

## Or delete everything
az group delete -n rg-ml-<alias> -y --no-wait