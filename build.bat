cmake ./
@echo off
echo search data.txt!
if exist "data.txt" (   
    echo data.txt is existence!
    exit
) 
echo data.txt is no find!
echo create data.txt...
if exist "train_data" (
    python totxt.py
    if exist ./data.txt ( 
        echo create data.txt success!
        exit
    )  
    echo ( unknow error! )
    exit
)
echo error no find ./train_data...

