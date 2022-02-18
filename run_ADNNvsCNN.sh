rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/data_1fs/gtdata/baseline/highway/*
python -u ADNNet_data.py            -pa_im          ./data/baseline/highway/input/     \
                                    -pa_gt          ./data/baseline/highway/groundtruth/  \
                                    -imgs_idx       831  \
                                    -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/data_1fs/gtdata/baseline/highway/  \

rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/data_1fs/gtdata/baseline/pedestrians/*
python -u ADNNet_data.py            -pa_im          ./data/baseline/pedestrians/input/     \
                                    -pa_gt          ./data/baseline/pedestrians/groundtruth/  \
                                    -imgs_idx       484  \
                                    -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/data_1fs/gtdata/baseline/pedestrians/  \

rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNvsCNN/CNNet_v14/network/baseline/*
python -u CNNet_train_v14.py    -train_data 2   /home/cqzhao/projects/matrix/data_UBgS_ADNNet/data_1fs/gtdata/baseline/highway/  \
                                                /home/cqzhao/projects/matrix/data_UBgS_ADNNet/data_1fs/gtdata/baseline/pedestrians/  \
                                -gpuid          0   \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNvsCNN/CNNet_v14/network/baseline/ \

rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNvsCNN/CNNet_v14/detectfg/baseline/office/*
python -u CNNet_detect_v14.py   -pa_net         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNvsCNN/CNNet_v14/network/baseline/ \
                                -gpuid          0 \
                                -idx_net        59  \
                                -pa_im          ./data/baseline/office/input/ \
                                -pa_gt          ./data/baseline/office/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNvsCNN/CNNet_v14/detectfg/baseline/office/  \

rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNvsCNN/ADNNet/network/baseline/*
python -u ADNNet_train.py       -train_data 2   /home/cqzhao/projects/matrix/data_UBgS_ADNNet/data_1fs/gtdata/baseline/highway/  \
                                                /home/cqzhao/projects/matrix/data_UBgS_ADNNet/data_1fs/gtdata/baseline/pedestrians/  \
                                -gpuid          0   \
                                -epochnum       60     \
                                -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNvsCNN/ADNNet/network/baseline/ \

rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNvsCNN/ADNNet/detectfg/baseline/office/*
python -u ADNNet_detect.py      -pa_net         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNvsCNN/ADNNet/network/baseline/ \
                                -gpuid          0 \
                                -idx_net        59  \
                                -pa_im          ./data/baseline/office/input/ \
                                -pa_gt          ./data/baseline/office/groundtruth/ \
                                -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNvsCNN/ADNNet/detectfg/baseline/office/  \

 
