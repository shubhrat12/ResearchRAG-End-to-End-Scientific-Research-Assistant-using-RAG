# LangChainGPT Dataset Extraction and Setup Script
# This script extracts and sets up the CORD-19 and PubLayNet datasets for transfer learning

# Display header
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "LangChainGPT Dataset Extraction and Setup" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# Define paths
$CORD19_ZIP = "archive (4).zip"
$PUBLAYNET_ZIP = "PubLayNet-master.zip"
$CORD19_TARGET_DIR = "data/training_datasets/cord19"
$CORD19_DOCUMENT_PARSES = "$CORD19_TARGET_DIR/document_parses"
$PUBLAYNET_TARGET_DIR = "data/training_datasets/publaynet"

function Create-Directory {
    param (
        [string]$path
    )
    
    if (-not (Test-Path $path)) {
        Write-Host "Creating directory: $path" -ForegroundColor Yellow
        New-Item -Path $path -ItemType Directory -Force | Out-Null
        Write-Host "✓ Created directory: $path" -ForegroundColor Green
    } else {
        Write-Host "✓ Directory already exists: $path" -ForegroundColor Green
    }
}

function Extract-Archive {
    param (
        [string]$zipFile,
        [string]$destination,
        [string]$description
    )
    
    if (-not (Test-Path $zipFile)) {
        Write-Host "ERROR: Could not find $description ZIP file: $zipFile" -ForegroundColor Red
        return $false
    }
    
    Write-Host "Extracting $description ($zipFile) to $destination..." -ForegroundColor Yellow
    try {
        Expand-Archive -Path $zipFile -DestinationPath $destination -Force
        Write-Host "✓ Successfully extracted $description" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "ERROR: Failed to extract $description" -ForegroundColor Red
        return $false
    }
}

function Count-Files {
    param (
        [string]$directory
    )
    
    if (-not (Test-Path $directory)) {
        return 0
    }
    
    $count = (Get-ChildItem -Path $directory -File | Measure-Object).Count
    return $count
}

function Fix-Directory-Structure {
    param (
        [string]$rootDir,
        [string]$expectedSubDir,
        [string]$description
    )
    
    Write-Host "Checking $description directory structure..." -ForegroundColor Yellow
    
    # Check if the expected directory exists
    if (-not (Test-Path "$rootDir/$expectedSubDir")) {
        # Look for similar directories that might contain our data
        $possibleSubDirs = Get-ChildItem -Path $rootDir -Directory -Recurse | Where-Object { 
            $_.Name -like "*$expectedSubDir*" -or $_.FullName -like "*$expectedSubDir*"
        }
        
        if ($possibleSubDirs.Count -gt 0) {
            Write-Host "Found potential match for $expectedSubDir" -ForegroundColor Yellow
            foreach ($dir in $possibleSubDirs) {
                Write-Host "  - $($dir.FullName)" -ForegroundColor Yellow
            }
            
            # Use the first match
            $sourceDir = $possibleSubDirs[0].FullName
            Write-Host "Moving files from $sourceDir to $rootDir/$expectedSubDir" -ForegroundColor Yellow
            
            # Create target directory
            Create-Directory "$rootDir/$expectedSubDir"
            
            # Move files
            Get-ChildItem -Path $sourceDir -Recurse | Move-Item -Destination "$rootDir/$expectedSubDir" -Force
            
            Write-Host "✓ Fixed directory structure for $description" -ForegroundColor Green
            return $true
        } else {
            Write-Host "ERROR: Could not find a matching directory for $expectedSubDir" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "✓ Directory structure is correct for $description" -ForegroundColor Green
        return $true
    }
}

# Step 1: Create directory structure
Write-Host "Step 1: Creating directory structure..." -ForegroundColor Cyan
Create-Directory $CORD19_TARGET_DIR
Create-Directory $CORD19_DOCUMENT_PARSES
Create-Directory "$CORD19_DOCUMENT_PARSES/pdf_json"
Create-Directory "$CORD19_DOCUMENT_PARSES/pmc_json"
Create-Directory $PUBLAYNET_TARGET_DIR
Write-Host ""

# Step 2: Extract datasets
Write-Host "Step 2: Extracting datasets..." -ForegroundColor Cyan
$cord19Extracted = Extract-Archive -zipFile $CORD19_ZIP -destination $CORD19_TARGET_DIR -description "CORD-19 dataset"
$publaynetExtracted = Extract-Archive -zipFile $PUBLAYNET_ZIP -destination $PUBLAYNET_TARGET_DIR -description "PubLayNet dataset"

if (-not ($cord19Extracted -and $publaynetExtracted)) {
    Write-Host "ERROR: Failed to extract one or both datasets. Exiting script." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Fix directory structure if needed
Write-Host "Step 3: Verifying and fixing directory structure..." -ForegroundColor Cyan

# Check for document_parses directory
if (-not (Test-Path "$CORD19_DOCUMENT_PARSES/pdf_json") -or -not (Test-Path "$CORD19_DOCUMENT_PARSES/pmc_json")) {
    # Look for document_parses in extracted files
    $documentParsesDir = Get-ChildItem -Path $CORD19_TARGET_DIR -Directory -Recurse | Where-Object {
        $_.Name -eq "document_parses" -or $_.Name -like "*document_parses*"
    } | Select-Object -First 1
    
    if ($null -ne $documentParsesDir) {
        Write-Host "Found document_parses directory: $($documentParsesDir.FullName)" -ForegroundColor Yellow
        
        # Check for pdf_json and pmc_json directories
        $pdfJsonDir = Get-ChildItem -Path $documentParsesDir.FullName -Directory | Where-Object {
            $_.Name -eq "pdf_json" -or $_.Name -like "*pdf_json*"
        } | Select-Object -First 1
        
        $pmcJsonDir = Get-ChildItem -Path $documentParsesDir.FullName -Directory | Where-Object {
            $_.Name -eq "pmc_json" -or $_.Name -like "*pmc_json*"
        } | Select-Object -First 1
        
        if ($null -ne $pdfJsonDir) {
            Write-Host "Found pdf_json directory: $($pdfJsonDir.FullName)" -ForegroundColor Yellow
            Write-Host "Moving pdf_json files to correct location..." -ForegroundColor Yellow
            Get-ChildItem -Path $pdfJsonDir.FullName -File -Filter "*.json" | Copy-Item -Destination "$CORD19_DOCUMENT_PARSES/pdf_json/" -Force
            Write-Host "✓ Moved pdf_json files" -ForegroundColor Green
        }
        
        if ($null -ne $pmcJsonDir) {
            Write-Host "Found pmc_json directory: $($pmcJsonDir.FullName)" -ForegroundColor Yellow
            Write-Host "Moving pmc_json files to correct location..." -ForegroundColor Yellow
            Get-ChildItem -Path $pmcJsonDir.FullName -File -Filter "*.json" | Copy-Item -Destination "$CORD19_DOCUMENT_PARSES/pmc_json/" -Force
            Write-Host "✓ Moved pmc_json files" -ForegroundColor Green
        }
    } else {
        Write-Host "WARNING: Could not find document_parses directory. Dataset structure may be incorrect." -ForegroundColor Yellow
    }
}

# Check for metadata.csv
$metadataFile = Get-ChildItem -Path $CORD19_TARGET_DIR -File -Recurse -Filter "metadata.csv" | Select-Object -First 1
if ($null -ne $metadataFile) {
    Write-Host "Found metadata.csv: $($metadataFile.FullName)" -ForegroundColor Yellow
    if ($metadataFile.DirectoryName -ne $CORD19_TARGET_DIR) {
        Write-Host "Moving metadata.csv to correct location..." -ForegroundColor Yellow
        Copy-Item -Path $metadataFile.FullName -Destination "$CORD19_TARGET_DIR/metadata.csv" -Force
        Write-Host "✓ Moved metadata.csv to $CORD19_TARGET_DIR" -ForegroundColor Green
    } else {
        Write-Host "✓ metadata.csv is already in the correct location" -ForegroundColor Green
    }
} else {
    Write-Host "WARNING: Could not find metadata.csv file." -ForegroundColor Yellow
}

# Verify PubLayNet structure
Fix-Directory-Structure -rootDir $PUBLAYNET_TARGET_DIR -expectedSubDir "PubLayNet" -description "PubLayNet"

Write-Host ""

# Step 4: Verify extraction
Write-Host "Step 4: Verifying extraction..." -ForegroundColor Cyan
$pdfJsonCount = Count-Files "$CORD19_DOCUMENT_PARSES/pdf_json"
$pmcJsonCount = Count-Files "$CORD19_DOCUMENT_PARSES/pmc_json"
Write-Host "Found $pdfJsonCount files in pdf_json directory" -ForegroundColor Yellow
Write-Host "Found $pmcJsonCount files in pmc_json directory" -ForegroundColor Yellow

if ($pdfJsonCount -eq 0 -and $pmcJsonCount -eq 0) {
    Write-Host "ERROR: No JSON files found in the document_parses directories. Extraction may have failed." -ForegroundColor Red
    exit 1
}

$publaynetFileCount = Count-Files $PUBLAYNET_TARGET_DIR
Write-Host "Found $publaynetFileCount files in the PubLayNet directory" -ForegroundColor Yellow
Write-Host ""

# Step 5: Run preparation script
Write-Host "Step 5: Running data preparation script..." -ForegroundColor Cyan
Write-Host "Executing: python prepare_training_data.py --cord19-samples 5000 --verbose" -ForegroundColor Yellow
$PrepResult = $null
$PrepResult = & python prepare_training_data.py --cord19-samples 5000 --verbose
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to run preparation script." -ForegroundColor Red
    exit 1
}
Write-Host "✓ Successfully ran preparation script" -ForegroundColor Green

# Final summary
Write-Host ""
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "Dataset Extraction and Setup Summary" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "CORD-19 files extracted: $($pdfJsonCount + $pmcJsonCount) JSON files" -ForegroundColor Green
Write-Host "PubLayNet files extracted: $publaynetFileCount files" -ForegroundColor Green
Write-Host "Datasets prepared for 5000 samples" -ForegroundColor Green
Write-Host ""
Write-Host "You can now proceed to run the transfer learning pipeline:" -ForegroundColor Cyan
Write-Host ".\run_transfer_learning.bat" -ForegroundColor Yellow
Write-Host "=======================================================" -ForegroundColor Cyan 