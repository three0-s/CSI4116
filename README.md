# CSI4116 Project3

Please download the dataset and upload it to your project directory

https://drive.google.com/file/d/1l-i9-Z0pNVynAFhyibwAkXSuLrMWiKm2/view?usp=sharing

## Overview

Please organize the project as follows:

```
$PRJ_ROOT/
    models.py
    submisson.py
    train.py
    datasets.py 
    datasets/
        test_subm.csv
        train_anno.csv
        images/
            test_0001.jpg
            ...
            test_0250.jpg
            train_0001.jpg
            ...
            train_2750.jpg
```

## How to run

For training, please run train.py with the defined arguments:
```
    python train.py --lr {find_proper_lr_value} --optim_type {adam_and_adamw_are_given_others_are_ok} --arch_ver {ver1} --ver_name {default}
```

For testing, please run submission.py
```
    python submission.py --arch_ver {arch_ver} --ckpt_path {path_to_your_trained_model}/ckpt/ep30.pt --num_crop {1/5/10} 
```

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--lr', default='1e-4')
    argparser.add_argument('--optim_type', default='adam')
    argparser.add_argument('--arch_ver', default='ver1')
    argparser.add_argument('--freeze', action='store_true')
    argparser.add_argument('--ver_name', default="")
    args = argparser.parse_args()