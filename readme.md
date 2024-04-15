Surface and Edge Detection for Primitive Fitting of Point Clouds
SIGGRAPH 2023 conference proceedings

## install
following [parsenet](https://github.com/Hippogriff/parsenet-codebase#installation) install cmds


## data prepare

- please download our dataset from [Onedrive](https://1drv.ms/f/s!AkbsfT9Y3igj3Hl9nmpQZsh7Vv5J?e=yOTZfe) or from [Baiduyun](https://pan.baidu.com/s/1apCmf8Xa_rXyRdWl4ybJpg?pwd=meta) (password is *meta*), and load all files of folder *sed_net_data* to *data* 
- please download parsenet datasets from [Onedrive](https://1drv.ms/f/s!AkbsfT9Y3igj3Hr1YQHzC8V0rO2-?e=XfwcSe) or from [Baiduyun](https://pan.baidu.com/s/16fggrr-qQRc2yu6ECQNaoA) (password is *meta*), and load all files of folder *parsenet* to *data_parsenet* 
- you can download our pretained models from [Onedrive](https://1drv.ms/f/s!AkbsfT9Y3igj3Hjl96WnhBMTAsWP?e=Akj76R) or from [Baiduyun](https://pan.baidu.com/s/1rMMD_0VaOGTmpMcIozjp3Q) (password is *meta*), and load all weights of folder *ckpts* to *ckpts*


## train & test
- test model with normal

```python 
python generate_predictions_aug.py configs/config_SEDNet_normal.yml NoSave no_multi_vote no_fold5drop
```

- train model with normal

```python 
python train_sed_net.py configs/config_SEDNet_normal.yml
```

## mesh creation
arg2mesh/arg2mesh.py 

## reference
1. [parsenet](https://github.com/Hippogriff/parsenet-codebase)
2. [hpnet](https://github.com/SimingYan/HPNet)

## Cite
@inproceedings{li2023surface,
  title={Surface and edge detection for primitive fitting of point clouds},
  author={Li, Yuanqi and Liu, Shun and Yang, Xinran and Guo, Jianwei and Guo, Jie and Guo, Yanwen},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  pages={1--10},
  year={2023}
}
