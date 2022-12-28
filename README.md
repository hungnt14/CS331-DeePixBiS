CS331-DeepPixBiS - DEEP PIXEL-WISE BINARY SUPERVISION FOR FACE ANTI SPOOFING

# Info
Course: CS331.N11.KHTN

Instructor: PhD. Tien-Dung Mai

Students:
| Name | Student ID |
|-|-|-|
| Lê Xuân Tùng | 20520347 |
| Lê Phước Vĩnh Linh | 20521531 |
| Nguyễn Tiến Hưng | 20520198 |

# Dataset:
We use the [Zalo AI Challenge 2022 - Liveness Detection task](https://challenge.zalo.ai/portal/liveness-detection) dataset to benchmark DeepPixBiS model [1]. First we hand-labelled 2 public test datasets. Then, we extracted frames using 2 methods: median frame and skip frames (1 second) from videos in this dataset. 

You can download the original dataset on [competition portal](https://challenge.zalo.ai/portal/liveness-detection) and you our provided scripts to extract frames (see `preprocess_scripts` folder). Make sure that your `original_data` folder has below structure to run our pre-process scripts properly.

```
.
├── groundtruths
└── videos
    ├── public_test_1/
    ├── public_test_2/
    └── train/
```

Or you can download our pre-processed data [here](https://drive.google.com/file/d/1G7zaXwnVBvrK7EyA7OvHT_lZkqLm-Vks/view?usp=share_link)

# Instruction
- Install require libraries:
```
pip3 install -r requirements.txt
```
- See provided example scripts to train and test (`scripts` folder)
  
- You can download our weights [here](https://drive.google.com/drive/folders/15ryxLKBN83_QHq9vTYVj_twSLgmHn_zH?usp=share_link)


# Reference
[1] [Deep Pixel-wise Binary Supervision for Face Presentation Attack Detection](https://arxiv.org/abs/1907.04047)
- Most of the code in this project were took from [deep-pix-bis-pad.pytorch](https://github.com/voqtuyen/deep-pix-bis-pad.pytorch) with some changes.