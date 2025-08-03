#!/usr/bin/env python3
"""
GIF Optimization Script
This script helps optimize GIF files for better web performance.
"""

import os
import sys
from pathlib import Path

def check_gif_file(gif_path):
    """Check if the GIF file exists and get its size."""
    if not os.path.exists(gif_path):
        print(f"❌ Error: GIF file not found at {gif_path}")
        return False
    
    file_size = os.path.getsize(gif_path)
    size_mb = file_size / (1024 * 1024)
    print(f"📁 GIF file found: {gif_path}")
    print(f"📊 Current size: {file_size:,} bytes ({size_mb:.2f} MB)")
    return True

def suggest_optimizations(gif_path):
    """Suggest optimization techniques for the GIF."""
    file_size = os.path.getsize(gif_path)
    size_mb = file_size / (1024 * 1024)
    
    print("\n🔧 Optimization Suggestions:")
    print("=" * 50)
    
    if size_mb > 1.0:
        print("⚠️  GIF is quite large (>1MB). Consider:")
        print("   • Reducing frame count")
        print("   • Lowering color palette (256 colors or less)")
        print("   • Reducing dimensions if possible")
        print("   • Using a video format (MP4/WebM) instead")
    
    print("\n📋 Manual Optimization Steps:")
    print("1. Use online tools like:")
    print("   • ezgif.com (free online GIF optimizer)")
    print("   • gifs.com (optimize and compress)")
    print("   • tinypng.com (image compression)")
    
    print("\n2. Use command-line tools:")
    print("   • gifsicle: gifsicle -O3 --lossy=80 input.gif -o output.gif")
    print("   • ImageMagick: convert input.gif -layers optimize output.gif")
    
    print("\n3. Consider alternatives:")
    print("   • Convert to MP4/WebM for better compression")
    print("   • Use CSS animations for simple effects")
    print("   • Use SVG animations for vector graphics")
    
    print(f"\n4. Target size: Aim for <500KB for optimal web performance")
    print(f"   Current: {size_mb:.2f}MB → Target: <0.5MB")

def create_optimization_script():
    """Create a batch script for Windows users to optimize the GIF."""
    script_content = """@echo off
echo GIF Optimization Script
echo =====================

echo.
echo Current GIF size:
dir "assets\\Title_Glitch_Transperent_0108.gif"

echo.
echo Optimization options:
echo 1. Use online tools (recommended for beginners)
echo    - Go to ezgif.com
echo    - Upload your GIF
echo    - Use the optimize tool
echo    - Download optimized version
echo.
echo 2. Install and use gifsicle (advanced)
echo    - Download from: https://eternallybored.org/misc/gifsicle/
echo    - Run: gifsicle -O3 --lossy=80 assets\\Title_Glitch_Transperent_0108.gif -o assets\\Title_Glitch_Transperent_0108_optimized.gif
echo.
echo 3. Use ImageMagick (advanced)
echo    - Install ImageMagick
echo    - Run: magick assets\\Title_Glitch_Transperent_0108.gif -layers optimize assets\\Title_Glitch_Transperent_0108_optimized.gif
echo.
pause
"""
    
    with open("optimize_gif.bat", "w") as f:
        f.write(script_content)
    
    print("✅ Created optimize_gif.bat - run this file for optimization instructions")

def main():
    gif_path = "assets/Title_Glitch_Transperent_0108.gif"
    
    print("🎬 GIF Performance Optimizer")
    print("=" * 40)
    
    if not check_gif_file(gif_path):
        sys.exit(1)
    
    suggest_optimizations(gif_path)
    create_optimization_script()
    
    print("\n✅ Optimization analysis complete!")
    print("💡 Try the optimizations above to improve loading speed.")

if __name__ == "__main__":
    main() 