# show databases;

# create database UOBtests;
# show databases like 'UOBtests';

##for method 1：
# show engines;
# drop table `UOBtests`.`result_of SD_STT`;

CREATE TABLE `UOBtests`.`result_of SD_STT`(
                                              `key` int NOT NULL AUTO_INCREMENT,
                                              `index` int NOT NULL,
                                              `audio_name` varchar(20)  NOT NULL,
                                              `start_time` float NOT NULL,
                                              `end_time` float NOT NULL,
                                              `duration` float,
                                              `label` varchar(20)  NOT NULL,
                                              `text` varchar(5000),
                                              `audio_path` varchar(100)  NOT NULL,
                                              `create_by` varchar(20),
                                              `create_date` date,
                                              `create_time` time,
                                              PRIMARY KEY (`key`,`index`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
    #A legacy of mysql, mysql utf8 can only support up to 3bytes length of character encoding, for some need to occupy 4bytes of text, mysql utf8 is not supported, you have to use utf8mb4
    #mb4 is most bytes 4, which uses 4 bytes to represent the full UTF-8
    #utf8mb4_bin: compile and store each character of the string in binary data, case sensitive, and can store the binary content
  AUTO_INCREMENT=1 ;#Specify the initial value of the self-increment: 1



desc `UOBtests`.`result_of SD_STT`;
select * from `UOBtests`.`result_of SD_STT`;

