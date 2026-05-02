@echo off
REM LangChainGPT Dataset Extraction Script
REM This script extracts CORD-19 and PubLayNet datasets to the correct locations

echo ==========================================
echo LangChainGPT Dataset Extraction
echo ==========================================

REM Create directories
echo Creating directories...
mkdir data\training_datasets\cord19\document_parses\pdf_json
mkdir data\training_datasets\cord19\document_parses\pmc_json
mkdir data\training_datasets\publaynet
echo Directories created.

REM Extract CORD-19 dataset
echo.
echo Extracting CORD-19 dataset...
powershell -Command "Expand-Archive -Path 'archive (4).zip' -DestinationPath 'data\training_datasets\cord19' -Force"
echo CORD-19 dataset extracted.

REM Extract PubLayNet dataset
echo.
echo Extracting PubLayNet dataset...
powershell -Command "Expand-Archive -Path 'PubLayNet-master.zip' -DestinationPath 'data\training_datasets\publaynet' -Force"
echo PubLayNet dataset extracted.

REM Fix structure if needed
echo.
echo Checking and fixing directory structure...

REM Check for metadata.csv
echo Looking for metadata.csv...
powershell -Command "$metadataFile = Get-ChildItem -Path 'data\training_datasets\cord19' -Recurse -Filter 'metadata.csv' | Select-Object -First 1; if ($metadataFile) { Copy-Item -Path $metadataFile.FullName -Destination 'data\training_datasets\cord19\metadata.csv' -Force; Write-Host 'Metadata file copied to root directory.' }"

REM Check for document_parses structure
echo Checking document_parses structure...
powershell -Command "$docParsesDir = Get-ChildItem -Path 'data\training_datasets\cord19' -Recurse -Directory | Where-Object { $_.Name -eq 'document_parses' } | Select-Object -First 1; if ($docParsesDir) { $pdfJsonDir = Get-ChildItem -Path $docParsesDir.FullName -Directory | Where-Object { $_.Name -eq 'pdf_json' } | Select-Object -First 1; if ($pdfJsonDir) { Get-ChildItem -Path $pdfJsonDir.FullName -Filter '*.json' | ForEach-Object { Copy-Item -Path $_.FullName -Destination 'data\training_datasets\cord19\document_parses\pdf_json\' -Force }; Write-Host 'PDF JSON files copied.' }; $pmcJsonDir = Get-ChildItem -Path $docParsesDir.FullName -Directory | Where-Object { $_.Name -eq 'pmc_json' } | Select-Object -First 1; if ($pmcJsonDir) { Get-ChildItem -Path $pmcJsonDir.FullName -Filter '*.json' | ForEach-Object { Copy-Item -Path $_.FullName -Destination 'data\training_datasets\cord19\document_parses\pmc_json\' -Force }; Write-Host 'PMC JSON files copied.' } }"

echo.
echo Dataset extraction completed. Running data preparation script:
echo.
echo python prepare_training_data.py --cord19-samples 5000 --verbose
python prepare_training_data.py --cord19-samples 5000 --verbose

echo.
echo ==========================================
echo Dataset preparation completed
echo ==========================================
echo.
echo You can now run the transfer learning:
echo .\run_transfer_learning.bat 