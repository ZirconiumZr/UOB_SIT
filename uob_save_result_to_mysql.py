from uob_dbConnect import connectDB
import time
import pandas as pd
from init import (
    temSdPath
)

def dbInsert(finalDf):
    connection = connectDB
    resultDf=finalDf.fillna(0)
    with connection:
        with connection.cursor() as cursor:
        #first: store result data to TBL_AUDIO table
            # Create a new record
            sql1 = "INSERT INTO `TBL_AUDIO` ( `audio_name`,`path_orig`,`path_processed`,`create_date`,`create_time`) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(sql1, ('dental_malaya',temSdPath,temSdPath,time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S")))

        # connection is not autocommit by default. So you must commit to save your changes.
        connection.commit()
        audio_id_1=cursor.lastrowid
        
        #second: store result data to TBL_STT_RESULT table
        #from dataframe"output" to get the SD, STT and label results
            # Create a new record
            sql2 = "INSERT INTO `TBL_STT_RESULT` (`audio_id`, `slice_id`,`start_time`,`end_time`,`duration`,`text`,`speaker_label`,`save_path`,`create_date`,`create_time`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            for i in range(0,len(resultDf)-1):
                cursor.execute(sql2, (audio_id_1,resultDf.index[i], resultDf.starttime[i],resultDf.endtime[i],resultDf.duration[i],resultDf.text[i],resultDf.label[i],temSdPath,time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S")))

        # connection is not autocommit by default. So you must commit to save your changes.
        connection.commit()