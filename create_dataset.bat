@echo off
echo Running calculate_global_z_min_max.py...
python code/calculate_global_z_min_max.py
if %errorlevel% neq 0 (
    echo Script calculate_global_z_min_max.py failed. Exiting...
    pause
    exit /b %errorlevel%
)

echo Running generate_segments_pointcloud_greyscale.py...
python code/generate_segments_pointcloud_greyscale.py
if %errorlevel% neq 0 (
    echo Script generate_segments_pointcloud_greyscale.py failed.
    pause
    exit /b %errorlevel%
)

echo All scripts completed successfully.
pause
