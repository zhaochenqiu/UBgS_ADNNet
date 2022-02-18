
rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/baseline/*
python -u CNNet_train_pv3.py    -train_data 2   /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/baseline/highway/ \
                                                /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/baseline/pedestrians/  \
                                -gpuid          2 \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/baseline/ \







rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/baseline/office/*
python -u CNNet_detect_pv3.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/office/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/office/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/baseline/office/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/baseline/PETS2006/*
python -u CNNet_detect_pv3.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/PETS2006/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/PETS2006/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/baseline/PETS2006/  \


rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/baseline/highway/*
python -u CNNet_detect_pv3.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/baseline/highway/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/baseline/pedestrians/*
python -u CNNet_detect_pv3.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/pedestrians/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/pedestrians/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/baseline/pedestrians/  \














