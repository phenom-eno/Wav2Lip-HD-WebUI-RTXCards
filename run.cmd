set /p USERID=<USERID.txt
call C:\Users\%USERID%\anaconda3\Scripts\activate.bat C:\Users\%USERID%\anaconda3\envs\wav2lip-ui

python ui.py

call conda deactivate