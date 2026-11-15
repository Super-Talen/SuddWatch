<#
SuddWatch - deploy the FastAPI container to Azure Container Apps (Azure for Students).

Prerequisites: Azure CLI (az login done), Docker Desktop running, run from the repository root.
Usage:
    powershell -ExecutionPolicy Bypass -File deploy\azure_deploy.ps1             # first deployment
    powershell -ExecutionPolicy Bypass -File deploy\azure_deploy.ps1 -Tag 1.0.1  # push a new image version

Notes from the first deployment (2026-11-15):
  * Azure for Students only allows a few regions; southafricanorth is the closest allowed one to South Sudan.
  * ACR Tasks (az acr build) is blocked on student subscriptions, so the image is built locally and pushed.
  * The Container Apps environment is an "express" environment, which does not support managed-identity
    registry pulls, so the registry admin credential is used; Azure stores it as an app secret.
  * min-replicas 0: the app scales to zero when idle, so it uses almost no credit between requests.
#>
param(
    [string]$Tag = "1.0.0",
    [string]$ResourceGroup = "suddwatch-rg",
    [string]$Location = "southafricanorth",
    [string]$Registry = "suddwatchacr8914",
    [string]$Environment = "suddwatch-env",
    [string]$App = "suddwatch-api"
)
$ErrorActionPreference = "Stop"
$image = "$Registry.azurecr.io/suddwatch-api:$Tag"

foreach ($p in "Microsoft.App", "Microsoft.OperationalInsights", "Microsoft.ContainerRegistry") {
    az provider register -n $p --wait
}
az group create -n $ResourceGroup -l $Location --tags project=SuddWatch | Out-Null
if (-not (az acr show -n $Registry -g $ResourceGroup --query name -o tsv 2>$null)) {
    az acr create -g $ResourceGroup -n $Registry --sku Basic -l $Location --admin-enabled true | Out-Null
}

docker build --platform linux/amd64 -t $image .
az acr login -n $Registry
docker push $image

if (az containerapp show -g $ResourceGroup -n $App --query name -o tsv 2>$null) {
    az containerapp update -g $ResourceGroup -n $App --image $image | Out-Null
} else {
    if (-not (az containerapp env show -g $ResourceGroup -n $Environment --query name -o tsv 2>$null)) {
        az containerapp env create -g $ResourceGroup -n $Environment -l $Location | Out-Null
    }
    $pw = az acr credential show -n $Registry --query "passwords[0].value" -o tsv
    az containerapp create -g $ResourceGroup -n $App --environment $Environment --image $image `
        --registry-server "$Registry.azurecr.io" --registry-username $Registry --registry-password $pw `
        --ingress external --target-port 8000 --cpu 0.5 --memory 1.0Gi --min-replicas 0 --max-replicas 1 `
        --tags project=SuddWatch | Out-Null
}
$fqdn = az containerapp show -g $ResourceGroup -n $App --query properties.configuration.ingress.fqdn -o tsv
Write-Host "Deployed: https://$fqdn/docs"
Invoke-RestMethod "https://$fqdn/health"

# Stop all charges when the course ends:  az group delete -n suddwatch-rg --yes
