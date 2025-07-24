#!/bin/bash

echo "🚀 Starting optimized deployment to Vercel..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

echo "📦 Installing dependencies..."
npm install

echo "🖼️  Running advanced image optimization..."
npm run optimize-images

echo "📄 Minifying HTML files..."
npm run minify-html

echo "🗺️  Generating sitemap..."
npm run generate-sitemap

echo "🔍 Running performance analysis..."
npm run analyze

echo "🚀 Deploying to Vercel..."
cd .. && vercel --prod

echo "✅ Deployment complete!"
echo "📊 Check your Vercel dashboard for performance metrics"
echo "🔗 Your site should now be much faster!" 