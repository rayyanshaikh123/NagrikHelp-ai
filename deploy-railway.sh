#!/bin/bash
# Railway Deployment Script
# Run this to deploy your AI server to Railway

echo "🚂 Deploying NagrikHelp AI to Railway"
echo "======================================"
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
    echo "✅ Railway CLI installed"
fi

# Login to Railway
echo "🔐 Logging into Railway..."
railway login

# Initialize project (if needed)
if [ ! -f ".railway" ]; then
    echo "🎯 Initializing Railway project..."
    railway init
fi

# Set environment variables
echo "⚙️  Setting environment variables..."
railway variables set CONFIDENCE_THRESHOLD=0.45
railway variables set ENABLE_YOLO=false
railway variables set ENABLE_CLIP=true
railway variables set ENABLE_RESNET=true

# Deploy
echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Your AI server will be available at:"
railway domain
echo ""
echo "📊 View logs:"
echo "   railway logs"
echo ""
echo "🔍 Check status:"
echo "   railway status"
