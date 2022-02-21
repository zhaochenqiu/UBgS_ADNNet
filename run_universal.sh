
rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/universal/fgimgs/baseline/highway/*
python -u ADNNet_detect.py          -pa_net         ./network/ \
                                    -gpuid          0   \
                                    -idx_net        59  \
                                    -pa_im          ./data/baseline/highway/input/     \
                                    -pa_gt          ./data/baseline/highway/groundtruth/  \
                                    -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/universal/fgimgs/baseline/highway/  \

rm -rf /home/cqzhao/projects/matrix/data_UBgS_ADNNet/universal/bayfg/baseline/highway/*
python -u Bayesian_refine.py        -gpuid          0   \
                                    -imgs_idx       -1  \
                                    -pa_im          ./data/baseline/highway/input/     \
                                    -pa_gt          ./data/baseline/highway/groundtruth/  \
                                    -pa_fg          /home/cqzhao/projects/matrix/data_UBgS_ADNNet/universal/fgimgs/baseline/highway/  \
                                    -pa_out         /home/cqzhao/projects/matrix/data_UBgS_ADNNet/universal/bayfg/baseline/highway/   \


