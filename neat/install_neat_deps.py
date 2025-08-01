"""
Install NEAT dependencies
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install required packages for NEAT"""
    packages = [
        "numpy>=1.19.0",
        "matplotlib>=3.3.0", 
        "networkx>=2.5",
        "imageio>=2.9.0"
    ]
    
    print("Installing NEAT dependencies...")
    
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
    
    print("\nDependency installation completed!")
    print("You can now run:")
    print("  python test_neat.py")
    print("  python train_neat.py --test")

if __name__ == "__main__":
    main() 