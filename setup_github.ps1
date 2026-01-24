# Script de configuration GitHub pour le projet
# Exécutez ce script dans PowerShell après avoir installé Git

Write-Host "=== Configuration GitHub ===" -ForegroundColor Cyan

# Vérifier si Git est installé
try {
    $gitVersion = git --version
    Write-Host "Git trouvé: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERREUR: Git n'est pas installé ou pas dans le PATH" -ForegroundColor Red
    Write-Host "Veuillez installer Git depuis https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Ou redémarrer votre terminal après l'installation" -ForegroundColor Yellow
    exit 1
}

# Demander les informations utilisateur
Write-Host "`nConfiguration Git:" -ForegroundColor Cyan
$userName = Read-Host "Entrez votre nom d'utilisateur GitHub"
$userEmail = Read-Host "Entrez votre email GitHub"

# Configurer Git
git config --global user.name "$userName"
git config --global user.email "$userEmail"

Write-Host "`nConfiguration Git terminée!" -ForegroundColor Green

# Vérifier si c'est un dépôt Git
if (Test-Path ".git") {
    Write-Host "`nDépôt Git détecté" -ForegroundColor Green
    
    # Vérifier les remotes existants
    $remotes = git remote -v
    if ($remotes) {
        Write-Host "`nRemotes existants:" -ForegroundColor Cyan
        Write-Host $remotes
    } else {
        Write-Host "`nAucun remote configuré" -ForegroundColor Yellow
        $addRemote = Read-Host "Voulez-vous ajouter un remote GitHub? (o/n)"
        if ($addRemote -eq "o" -or $addRemote -eq "O") {
            $repoUrl = Read-Host "Entrez l'URL de votre dépôt GitHub (ex: https://github.com/username/repo.git)"
            git remote add origin $repoUrl
            Write-Host "Remote 'origin' ajouté!" -ForegroundColor Green
        }
    }
} else {
    Write-Host "`nCe répertoire n'est pas un dépôt Git" -ForegroundColor Yellow
    $initRepo = Read-Host "Voulez-vous initialiser un dépôt Git? (o/n)"
    if ($initRepo -eq "o" -or $initRepo -eq "O") {
        git init
        Write-Host "Dépôt Git initialisé!" -ForegroundColor Green
        
        $addRemote = Read-Host "Voulez-vous ajouter un remote GitHub? (o/n)"
        if ($addRemote -eq "o" -or $addRemote -eq "O") {
            $repoUrl = Read-Host "Entrez l'URL de votre dépôt GitHub (ex: https://github.com/username/repo.git)"
            git remote add origin $repoUrl
            Write-Host "Remote 'origin' ajouté!" -ForegroundColor Green
        }
    }
}

Write-Host "`n=== Configuration terminée ===" -ForegroundColor Cyan
Write-Host "`nPour vous authentifier avec GitHub, vous pouvez:" -ForegroundColor Yellow
Write-Host "1. Utiliser GitHub Desktop (recommandé): https://desktop.github.com/" -ForegroundColor White
Write-Host "2. Utiliser GitHub CLI: gh auth login" -ForegroundColor White
Write-Host "3. Utiliser un Personal Access Token lors des push" -ForegroundColor White
Write-Host "   Créez un token ici: https://github.com/settings/tokens" -ForegroundColor White
