# Azure Container Apps Deployment Guide

This guide covers deploying the FastAPI + FAISS backend to Azure Container Apps with Blob Storage integration.

## Prerequisites

- Azure CLI installed and logged in
- Azure for Students subscription active
- Docker installed (for local testing)

## Step 1: Create Azure Resources

```bash
# Set variables (customize these)
RESOURCE_GROUP="faiss-backend-rg"
LOCATION="centralindia"
ACR_NAME="faissacr123"  # Must be globally unique
STORAGE_ACCOUNT="faissbackendstorage123"  # Must be globally unique
CONTAINER_APP_ENV="faiss-env"
CONTAINER_APP_NAME="faiss-backend-app"
CONTAINER_NAME="faiss-indexes"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create storage account
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

# Create blob container
az storage container create \
  --account-name $STORAGE_ACCOUNT \
  --name $CONTAINER_NAME \
  --auth-mode login

# Create Azure Container Registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Basic \
  --location $LOCATION

# Create Container Apps environment
az containerapp env create \
  --name $CONTAINER_APP_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION
```

## Step 2: Build and Push Docker Image

```bash
# Login to ACR
az acr login --name $ACR_NAME

# Build and push image
docker build -t $ACR_NAME.azurecr.io/faiss-backend:latest .
docker push $ACR_NAME.azurecr.io/faiss-backend:latest
```

## Step 3: Deploy Container App

### Option A: Using Connection String (Quick Setup)

```bash
# Get storage connection string
CONNECTION_STRING=$(az storage account show-connection-string \
  --resource-group $RESOURCE_GROUP \
  --name $STORAGE_ACCOUNT \
  --query connectionString \
  --output tsv)

# Deploy container app
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_APP_ENV \
  --image $ACR_NAME.azurecr.io/faiss-backend:latest \
  --target-port 8000 \
  --ingress external \
  --registry-server $ACR_NAME.azurecr.io \
  --cpu 1.0 --memory 2.0Gi \
  --secrets azure-storage-connection-string="$CONNECTION_STRING" \
  --env-vars \
    AZURE_STORAGE_CONNECTION_STRING=secretref:azure-storage-connection-string \
    AZURE_BLOB_CONTAINER=$CONTAINER_NAME \
    GEMINI_API_KEY=secretref:gemini-api-key \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000
```

### Option B: Using Managed Identity (Recommended)

```bash
# Create container app with managed identity
az containerapp create \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_APP_ENV \
  --image $ACR_NAME.azurecr.io/faiss-backend:latest \
  --target-port 8000 \
  --ingress external \
  --registry-server $ACR_NAME.azurecr.io \
  --cpu 1.0 --memory 2.0Gi \
  --assign-identity

# Get the container app's managed identity
IDENTITY_ID=$(az containerapp identity show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query principalId \
  --output tsv)

# Assign Storage Blob Data Contributor role to managed identity
az role assignment create \
  --assignee $IDENTITY_ID \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Storage/storageAccounts/$STORAGE_ACCOUNT"

# Update container app with managed identity settings
az containerapp update \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --set-env-vars \
    AZURE_STORAGE_ACCOUNT_URL=https://$STORAGE_ACCOUNT.blob.core.windows.net \
    AZURE_BLOB_CONTAINER=$CONTAINER_NAME \
    GEMINI_API_KEY=secretref:gemini-api-key \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000
```

## Step 4: Configure Secrets

```bash
# Add Gemini API key as secret
az containerapp secret set \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --secrets gemini-api-key="YOUR_GEMINI_API_KEY_HERE"
```

## Step 5: Verify Deployment

```bash
# Get the container app URL
FQDN=$(az containerapp show \
  --name $CONTAINER_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  --output tsv)

echo "Container App URL: https://$FQDN"

# Test health endpoint
curl https://$FQDN/health
```

## Step 6: Set Up CI/CD (Optional)

1. Create Azure service principal:

```bash
az ad sp create-for-rbac --name "faiss-backend-cicd" \
  --role contributor \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP" \
  --sdk-auth
```

2. Add the output JSON as `AZURE_CREDENTIALS` secret in GitHub repository settings.

3. Update the workflow file (`.github/workflows/deploy-acr.yml`) with your resource names.

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for embeddings | Yes |
| `AZURE_STORAGE_CONNECTION_STRING` | Storage account connection string (Option A) | No* |
| `AZURE_STORAGE_ACCOUNT_URL` | Storage account URL for managed identity (Option B) | No* |
| `AZURE_BLOB_CONTAINER` | Blob container name | No (defaults to faiss-indexes) |
| `LOG_LEVEL` | Logging level | No (defaults to INFO) |
| `API_HOST` | API host | No (defaults to 0.0.0.0) |
| `API_PORT` | API port | No (defaults to 8000) |

*Either connection string OR account URL must be provided for blob storage.

## Troubleshooting

### Container fails to start
- Check logs: `az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP`
- Verify environment variables are set correctly
- Ensure container image was pushed successfully

### Blob storage access fails
- Verify managed identity has correct role assignment
- Check storage account firewall settings
- Confirm container name exists

### Health check fails
- Verify target port (8000) matches container EXPOSE port
- Check if dependencies (FAISS, sentence-transformers) installed correctly
- Review application logs for import errors

### Performance issues
- Increase container CPU/memory: `--cpu 2.0 --memory 4.0Gi`
- Consider using larger Azure Container Apps SKU
- Monitor container metrics in Azure portal