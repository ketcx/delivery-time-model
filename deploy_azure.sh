#!/bin/bash

# Deploy to Azure Container Instances - Using Azure Container Registry Tasks (No local Docker needed)
# This script builds the image directly in Azure using ACR Tasks

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Azure Container Instance Deployment (No Docker Required) ===${NC}\n"

# Configuration
RESOURCE_GROUP="${RESOURCE_GROUP:-delivery-model-rg}"
LOCATION="${LOCATION:-eastus}"
ACR_NAME="${ACR_NAME:-deliverymodelacr}"
CONTAINER_NAME="${CONTAINER_NAME:-delivery-api}"
IMAGE_NAME="delivery-prediction-api"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Minimal resources for cost optimization
CPU_CORES="0.5"
MEMORY_GB="1.0"
PORT="8000"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  ACR Name: $ACR_NAME"
echo "  Container Name: $CONTAINER_NAME"
echo "  CPU: $CPU_CORES cores"
echo "  Memory: ${MEMORY_GB}GB"
echo ""

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo -e "${RED}Error: Azure CLI not found. Please install it first.${NC}"
    exit 1
fi

# Check if logged in
echo -e "${GREEN}[1/6] Checking Azure login...${NC}"
if ! az account show &> /dev/null; then
    echo -e "${YELLOW}Not logged in. Running 'az login'...${NC}"
    az login
fi

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "  Using subscription: $SUBSCRIPTION_ID"

# Register required providers
echo -e "\n${GREEN}[2/6] Registering Azure providers...${NC}"
echo "  Registering Microsoft.ContainerRegistry..."
az provider register --namespace Microsoft.ContainerRegistry --output none
echo "  Registering Microsoft.ContainerInstance..."
az provider register --namespace Microsoft.ContainerInstance --output none
echo "  ⏳ Waiting for providers to register (this may take 1-2 minutes)..."
sleep 30

# Create resource group
echo -e "\n${GREEN}[3/6] Creating resource group...${NC}"
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none
echo "  ✓ Resource group created/verified"

# Create Azure Container Registry
echo -e "\n${GREEN}[4/6] Creating Azure Container Registry...${NC}"
if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    az acr create \
        --name "$ACR_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --sku Basic \
        --admin-enabled true \
        --location "$LOCATION" \
        --output none
    echo "  ✓ ACR created"
else
    echo "  ✓ ACR already exists"
fi

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query passwords[0].value -o tsv)
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)

# Build image using Azure Container Registry Tasks (no local Docker needed!)
echo -e "\n${GREEN}[5/6] Building Docker image in Azure (using ACR Tasks)...${NC}"
echo "  Building: $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
az acr build \
    --registry "$ACR_NAME" \
    --image "$IMAGE_NAME:$IMAGE_TAG" \
    --file Dockerfile \
    .
echo "  ✓ Image built and pushed to ACR"

# Deploy to Azure Container Instance
echo -e "\n${GREEN}[6/6] Deploying to Azure Container Instances...${NC}"
az container create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
    --registry-login-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --cpu "$CPU_CORES" \
    --memory "$MEMORY_GB" \
    --os-type Linux \
    --dns-name-label "$CONTAINER_NAME-$(date +%s)" \
    --ports "$PORT" \
    --restart-policy OnFailure \
    --output none

echo "  ✓ Container deployed"

# Get endpoint URL
echo -e "\n${GREEN}========================================${NC}"
echo "  Getting endpoint information..."
FQDN=$(az container show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --query ipAddress.fqdn -o tsv)

API_URL="http://${FQDN}:${PORT}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}API Endpoint:${NC}"
echo "  $API_URL"
echo ""
echo -e "${YELLOW}API Documentation:${NC}"
echo "  $API_URL/docs"
echo ""
echo -e "${YELLOW}Test the API:${NC}"
echo "  curl $API_URL/health"
echo ""
echo "  # Make a prediction:"
echo '  curl -X POST '$API_URL'/predict \\'
echo '    -H "Content-Type: application/json" \\'
echo '    -d '"'"'{"distance_miles": 5.5, "time_of_day_hours": 17.5, "is_weekend": 0}'"'"
echo ""
echo -e "${YELLOW}View logs:${NC}"
echo "  az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""
echo -e "${YELLOW}Estimated monthly cost:${NC}"
echo "  • Container Instances (0.5 CPU, 1GB RAM): ~\$3-5 USD/month (pay-per-second)"
echo "  • Azure Container Registry (Basic): \$5 USD/month (50GB storage)"
echo "  • Total: ~\$8-10 USD/month"
echo ""
echo -e "${YELLOW}To stop the container and save costs:${NC}"
echo "  az container stop --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo ""
echo -e "${YELLOW}To delete everything:${NC}"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo ""
