cmake ./
@echo off
echo search data.txt!
if exist ./data.txt (    
    echo data.txt is existence!
) else (
    echo data.txt is no find!\n
    echo create data.txt...
    if exist ./train_data (
        python totxt.py
        if exist ./data.txt ( 
            echo create data.txt success!
        )
        else (
            echo unknow error!
        )
    ) else (
        echo error no find ./train_data...
    )
)

