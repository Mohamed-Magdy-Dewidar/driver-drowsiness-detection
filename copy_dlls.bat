@echo off
echo ✅ Copying OpenCV DLLs to build\Release...
xcopy "C:\opencv\build\x64\vc16\bin\*.dll" ".\build\Release\" /Y
xcopy "C:\opencv\build\x64\vc16\bin\*.dll" ".\build\Debug\" /Y
echo ✅ Done.