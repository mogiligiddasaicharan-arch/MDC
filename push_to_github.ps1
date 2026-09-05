# push_to_github.ps1 - Push project to GitHub

$repoName = "manufacturing-defect-classification"
$githubUser = Read-Host "Enter your GitHub username"

Write-Host "`nSetting up GitHub repository..." -ForegroundColor Cyan

# Check if git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Git is not installed. Download from https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

# Initialize git if not already
if (!(Test-Path ".git")) {
    Write-Host "Initializing git repository..." -ForegroundColor Yellow
    git init
    git branch -M main
}

# Create .gitignore if it doesn't exist
if (!(Test-Path ".gitignore")) {
    Write-Host "Creating .gitignore..." -ForegroundColor Yellow
    @"
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt

# IDEs
.idea/
.vscode/
*.swp
*.swo

# Large files (don't commit to git)
datasets/
checkpoints/
*.pth
*.pt
*.h5
*.onnx

# Logs
*.log
logs/
wandb/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Results
results/
outputs/
predictions/
"@ | Out-File -FilePath ".gitignore" -Encoding utf8
}

# Stage all files
Write-Host "`nStaging files..." -ForegroundColor Yellow

# Add source files
git add *.py
git add .gitignore
git add requirements.txt
if (Test-Path "README.md") { git add README.md }

# Check status
git status

# Commit
$commitMsg = Read-Host "`nEnter commit message (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($commitMsg)) {
    $commitMsg = "Initial commit: Manufacturing Defect Classification"
}

git commit -m "$commitMsg"

# Add remote and push
$remoteUrl = "https://github.com/$githubUser/$repoName.git"
Write-Host "`nRemote URL: $remoteUrl" -ForegroundColor Gray

# Check if remote exists
$remotes = git remote
if ($remotes -contains "origin") {
    Write-Host "Remote 'origin' already exists. Updating URL..." -ForegroundColor Yellow
    git remote set-url origin $remoteUrl
} else {
    git remote add origin $remoteUrl
}

Write-Host "`nPushing to GitHub..." -ForegroundColor Green
git push -u origin main

Write-Host "`nDone! Repository pushed to: $remoteUrl" -ForegroundColor Cyan
Write-Host "`nNOTE: Large files (datasets/, checkpoints/) are excluded via .gitignore"
Write-Host "To share model weights, use GitHub Releases or Git LFS." -ForegroundColor Yellow
