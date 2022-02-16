rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/gtdata/baseline/highway/*
python -u ADNNet_data.py            -pa_im          ./data/baseline/highway/input/     \
                                    -pa_gt          ./data/baseline/highway/groundtruth/  \
                                    -imgs_idx       831  931 1698   \
                                    -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/gtdata/baseline/highway/  \

rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNet/baseline/highway/*
python -u ADNNet_train.py           -train_data     1   /home/cqzhao/projects/matrix/data_UBgS_ADNNet/gtdata/baseline/highway/  \
                                    -gpuid          0   \
                                    -epochnum       60  \
                                    -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNet/baseline/highway/  \

rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/fgimgs/baseline/highway/*
python -u ADNNet_detect.py     -pa_net         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/ADNNet/baseline/highway/  \
                                    -gpuid          0   \
                                    -idx_net        59  \
                                    -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/input/     \
                                    -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/groundtruth/  \
                                    -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/fgimgs/baseline/highway/  \

rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/bayfg/baseline/highway/*
python -u Bayesian_refine.py        -gpuid          0   \
                                    -imgs_idx       -1  \
                                    -pa_im          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/input/     \
                                    -pa_gt          /home/cqzhao/dataset/dataset2014/dataset/baseline/highway/groundtruth/  \
                                    -pa_fg          /home/cqzhao/projects/matrix/data_UBgS_ADNNet/fgimgs/baseline/highway/  \
                                    -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/bayfg/baseline/highway/   \




