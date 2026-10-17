param([ValidateSet('cu128','cu130')][string]$Cuda = 'cu130')
$ErrorActionPreference = 'Stop'
$trainingPython = Join-Path $PSScriptRoot 'backend\.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $trainingPython)) { throw 'Create backend/.venv before installing GPU dependencies.' }
Write-Host 'Close the backend and any training workers before updating PyTorch.'
& $trainingPython -m pip install --upgrade --force-reinstall torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch CUDA installation failed. Check network access, Python wheel availability and NVIDIA driver compatibility.' }
& $trainingPython -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA unavailable: check NVIDIA driver and PyTorch build'; x=torch.ones(2,device='cuda'); print('GPU:', torch.cuda.get_device_name(0)); print('GPU tensor test:', (x+x).tolist())"
if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed. The application can still run on CPU.' }
Write-Host 'GPU is ready. Restart the backend to use it.'
