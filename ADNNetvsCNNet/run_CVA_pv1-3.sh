

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/baseline/*
python -u CNNet_train_pv1.py    -train_data 2   /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/baseline/highway/ \
                                                /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/baseline/pedestrians/  \
                                -gpuid          2 \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/baseline/ \


rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/baseline/*
python -u CNNet_train_pv2.py    -train_data 2   /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/baseline/highway/ \
                                                /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/baseline/pedestrians/  \
                                -gpuid          2 \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/baseline/ \


rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/baseline/*
python -u CNNet_train_pv3.py    -train_data 2   /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/baseline/highway/ \
                                                /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/baseline/pedestrians/  \
                                -gpuid          2 \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/baseline/ \


rm -rf /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/baseline/*
python -u ADNNet_train.py       -train_data 2   /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/baseline/highway/ \
                                                /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/baseline/pedestrians/  \
                                -gpuid          2 \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/baseline/ \







rm -rf /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/baseline/office/*
python -u ADNNet_detect_fast.py -pa_net         /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/office/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/office/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/baseline/office/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/baseline/PETS2006/*
python -u ADNNet_detect_fast.py -pa_net         /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/PETS2006/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/PETS2006/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/baseline/PETS2006/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/baseline/highway/*
python -u ADNNet_detect_fast.py -pa_net         /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/baseline/highway/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/baseline/pedestrians/*
python -u ADNNet_detect_fast.py -pa_net         /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/pedestrians/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/pedestrians/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/baseline/pedestrians/  \











rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/baseline/office/*
python -u CNNet_detect_pv1.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/office/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/office/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/baseline/office/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/baseline/PETS2006/*
python -u CNNet_detect_pv1.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/PETS2006/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/PETS2006/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/baseline/PETS2006/  \


rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/baseline/highway/*
python -u CNNet_detect_pv1.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/baseline/highway/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/baseline/pedestrians/*
python -u CNNet_detect_pv1.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/pedestrians/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/pedestrians/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/baseline/pedestrians/  \








rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/baseline/office/*
python -u CNNet_detect_pv2.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/office/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/office/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/baseline/office/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/baseline/PETS2006/*
python -u CNNet_detect_pv2.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/PETS2006/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/PETS2006/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/baseline/PETS2006/  \


rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/baseline/highway/*
python -u CNNet_detect_pv2.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/baseline/highway/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/baseline/pedestrians/*
python -u CNNet_detect_pv2.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/baseline/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/pedestrians/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/pedestrians/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/baseline/pedestrians/  \








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




























rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/dynamicBackground/*
python -u CNNet_train_pv1.py    -train_data 2   /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/dynamicBackground/fountain01/  \
                                                /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/dynamicBackground/overpass/  \
                                -gpuid          2 \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/dynamicBackground/ \



rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/dynamicBackground/*
python -u CNNet_train_pv2.py    -train_data 2   /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/dynamicBackground/fountain01/  \
                                                /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/dynamicBackground/overpass/  \
                                -gpuid          2 \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/dynamicBackground/ \



rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/dynamicBackground/*
python -u CNNet_train_pv3.py    -train_data 2   /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/dynamicBackground/fountain01/  \
                                                /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/dynamicBackground/overpass/  \
                                -gpuid          2 \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/dynamicBackground/ \


rm -rf /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/dynamicBackground/*
python -u ADNNet_train.py       -train_data 2   /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/dynamicBackground/fountain01/  \
                                                /home/cqzhao/projects/matrix/data_1fs/trainingdata/gtdata/dynamicBackground/overpass/  \
                                -gpuid          2 \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/dynamicBackground/ \





rm -rf /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/dynamicBackground/canoe/*
python -u ADNNet_detect_fast.py -pa_net         /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/canoe/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/canoe/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/dynamicBackground/canoe/  \


rm -rf /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/dynamicBackground/fountain02/*
python -u ADNNet_detect_fast.py -pa_net         /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain02/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain02/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/dynamicBackground/fountain02/  \


rm -rf /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/dynamicBackground/fountain01/*
python -u ADNNet_detect_fast.py -pa_net         /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/dynamicBackground/fountain01/  \


rm -rf /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/dynamicBackground/overpass/*
python -u ADNNet_detect_fast.py -pa_net         /home/cqzhao/projects/matrix/data_CVA/ADNNet/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/overpass/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/overpass/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/ADNNet/detectfg/dynamicBackground/overpass/  \








rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/dynamicBackground/fountain01/*
python -u CNNet_detect_pv1.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/dynamicBackground/fountain01/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/dynamicBackground/fountain02/*
python -u CNNet_detect_pv1.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain02/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain02/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/dynamicBackground/fountain02/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/dynamicBackground/overpass/*
python -u CNNet_detect_pv1.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/overpass/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/overpass/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/dynamicBackground/overpass/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/dynamicBackground/canoe/*
python -u CNNet_detect_pv1.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/canoe/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/canoe/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv1/detectfg/dynamicBackground/canoe/  \






rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/dynamicBackground/fountain01/*
python -u CNNet_detect_pv2.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/dynamicBackground/fountain01/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/dynamicBackground/fountain02/*
python -u CNNet_detect_pv2.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain02/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain02/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/dynamicBackground/fountain02/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/dynamicBackground/overpass/*
python -u CNNet_detect_pv2.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/overpass/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/overpass/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/dynamicBackground/overpass/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/dynamicBackground/canoe/*
python -u CNNet_detect_pv2.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/canoe/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/canoe/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv2/detectfg/dynamicBackground/canoe/  \







rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/dynamicBackground/fountain01/*
python -u CNNet_detect_pv3.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain01/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/dynamicBackground/fountain01/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/dynamicBackground/fountain02/*
python -u CNNet_detect_pv3.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain02/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/fountain02/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/dynamicBackground/fountain02/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/dynamicBackground/overpass/*
python -u CNNet_detect_pv3.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/overpass/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/overpass/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/dynamicBackground/overpass/  \

rm -rf /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/dynamicBackground/canoe/*
python -u CNNet_detect_pv3.py    -pa_net        /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/network/dynamicBackground/ \
                                -gpuid          2 \
                                -idx_net 59  \
                                -pa_im          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/canoe/input/ \
                                -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/dynamicBackground/canoe/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_CVA/CNNet_pv3/detectfg/dynamicBackground/canoe/  \




