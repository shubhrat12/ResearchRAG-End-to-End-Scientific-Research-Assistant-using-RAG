@echo off
REM LangChainGPT Project - Setup Transfer Learning Data
REM This script sets up symbolic links from prepared datasets to transfer learning expected locations

echo ==========================================
echo Setting up Transfer Learning Data Paths
echo ==========================================

REM Create required directories
if not exist data\training_datasets\cord19\section_classification mkdir data\training_datasets\cord19\section_classification
if not exist data\training_datasets\cord19\figure_detection mkdir data\training_datasets\cord19\figure_detection
if not exist data\training_datasets\cord19\reference_parsing mkdir data\training_datasets\cord19\reference_parsing

REM Create symbolic links for section classification data
echo Creating symbolic links for section classification data...
mklink data\training_datasets\cord19\section_classification\train.json data\transfer_learning\prepared\section_classification\train\data.json
mklink data\training_datasets\cord19\section_classification\val.json data\transfer_learning\prepared\section_classification\val\data.json
mklink data\training_datasets\cord19\section_classification\test.json data\transfer_learning\prepared\section_classification\test\data.json

REM Create symbolic links for figure detection data
echo Creating symbolic links for figure detection data...
mklink data\training_datasets\cord19\figure_detection\train.json data\transfer_learning\prepared\figure_detection\train\data.json
mklink data\training_datasets\cord19\figure_detection\val.json data\transfer_learning\prepared\figure_detection\val\data.json
mklink data\training_datasets\cord19\figure_detection\test.json data\transfer_learning\prepared\figure_detection\test\data.json

REM Create symbolic links for reference parsing data
echo Creating symbolic links for reference parsing data...
mklink data\training_datasets\cord19\reference_parsing\train.json data\transfer_learning\prepared\reference_parsing\train\data.json
mklink data\training_datasets\cord19\reference_parsing\val.json data\transfer_learning\prepared\reference_parsing\val\data.json
mklink data\training_datasets\cord19\reference_parsing\test.json data\transfer_learning\prepared\reference_parsing\test\data.json

echo ==========================================
echo Data setup complete! 
echo Ready to run transfer learning process.
echo ========================================== 