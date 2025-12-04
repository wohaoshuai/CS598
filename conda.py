import os

print(os.path.abspath("/"))          # System root: '/' (Linux/Mac) or 'C:\' (Windows)
print(os.environ.get('CONDA_PREFIX')) # Your conda env path: '/home/username/anaconda3/envs/myenv'
