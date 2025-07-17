@echo off
echo Running calculate_global_z_min_max.py...
python code/create_dataset/calculate_global_z_min_max.py
if %errorlevel% neq 0 (
    echo Script calculate_global_z_min_max.py failed. Exiting...
    pause
    exit /b %errorlevel%
)

echo Running generate_segments_pointcloud_greyscale.py...
python code/create_dataset/generate_segments_pointcloud_greyscale.py
if %errorlevel% neq 0 (
    echo Script generate_segments_pointcloud_greyscale.py failed.
    pause
    exit /b %errorlevel%
)

echo Running split_traing_validate_test_dataset.py...
python code/create_dataset/split_traing_validate_test_dataset.py
if %errorlevel% neq 0 (
    echo Script split_traing_validate_test_dataset.py failed.
    pause
    exit /b %errorlevel%
)

echo All scripts completed successfully, dataset created
pause
