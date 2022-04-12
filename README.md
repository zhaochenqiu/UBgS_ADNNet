# This is the implementation of paper "Universal Background Subtraction based on Arithmetic Distribution Neural Network"
## Abstract
We propose a universal background subtraction framework based on the Arithmetic Distribution Neural Network (ADNN) for learning the distributions of temporal pixels. In our ADNN model, the arithmetic distribution operations are utilized to introduce the arithmetic distribution layers, including the product distribution layer and the sum distribution layer. Furthermore, in order to improve the accuracy of the proposed approach, an improved Bayesian refinement model based on neighboring information, with a GPU implementation, is incorporated. In the forward pass and backpropagation of the proposed arithmetic distribution layers, histograms are considered as probability density functions rather than matrices. Thus, the proposed approach is able to utilize the probability information of the histogram and achieve promising results with a very simple architecture compared to traditional convolutional neural networks. Evaluations using standard benchmarks demonstrate the superiority of the proposed approach compared to state-of-the-art traditional and deep learning methods. To the best of our knowledge, this is the first method to propose network layers based on arithmetic distribution operations for learning distributions during background subtraction.

## Running environment 
Anaconda + Pytorch

## How to run the code
We provided a shell script `run_demo.sh` to run our code. It is a demo of using ADNNet (Arithmetic Distribution Neural Network) for background subtraction. Then, `run_universal.sh` provided a pre-trained model which is trained with less than 1\% of ground truth frames from the CDNet2014 dataset but tested for all videos from CDNet2014, LASIESTA, and SBMI2015 datasets. For more details, please check our paper. Moreover, `run_ADNNvsCNN.sh` proposed a comparison between the ADNNet and CNNet (convolutional neural network) in background subtraction, for more comparison results, please check our paper. In addition, folder `verification` provided the verification of the ADNNet. 

## Citation
----
Please cite our paper, If you use this code, please cite our paper:
            @ARTICLE{9749010,
              author={Zhao, Chenqiu and Hu, Kangkang and Basu, Anup},
              journal={IEEE Transactions on Image Processing}, 
              title={Universal Background Subtraction Based on Arithmetic Distribution Neural Network}, 
              year={2022},
              volume={31},
              number={},
              pages={2934-2949},
              doi={10.1109/TIP.2022.3162961}}
----
