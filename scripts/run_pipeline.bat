@echo off
echo Running the data processing pipeline...
python -m src.pipelines.build_master_dataset
echo Pipeline execution completed.