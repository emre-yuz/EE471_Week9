@echo off
REM Clean pose estimation runner - filters out MediaPipe warnings
for /f "delims=" %%i in ('python POSE_ESTIMATION.py %* 2^>nul') do echo %%i