# Visualize what you learn: A well-explainable joint-learning framework based on multi-view mammograms and associated reports

In this paper, we introduce a novel pre-training framework for label-efficient medical image recognition, which we refer to as the "**multiview-allowed** radiograph joint **exam-level** report" approach. Our proposed strategy "visualize what you learn" is designed to provide a comprehensive and easily interpretable visualization of the visual and textual features learned by deep learning models, thereby enabling developers to assess the depth of the model's understanding beyond its performance.

We evaluate the performance of our framework on various medical imaging datasets, including classification, segmentation, and localization tasks in both fine-tuning and zero-shot settings. Our results demonstrate that our proposed approach achieves high performance and label efficiency compared to existing state-of-the-art methods. Overall, our approach offers a promising direction for developing more robust and effective medical image recognition systems.


## Approach

[comment]: <> (* As illustrated in our workflow, we introduce A<sup>2</sup>I<sup>2</sup>, including abnormality-awareness &#40;Module-A<sup>2</sup>&#41; &#40;Fig.&#40;b&#41;&#41; consists of a visualization module and Module-I<sup>2</sup> &#40;Fig.&#40;c&#41;&#41; aims intra-mammogram &#40;multi-view&#41; and inter-modal learning.)
![A<sup>2</sup>I<sup>2</sup>](https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/model.png)

## Visualization

[comment]: <> (* As illustrated in our workflow, we introduce A<sup>2</sup>I<sup>2</sup>, including abnormality-awareness &#40;Module-A<sup>2</sup>&#41; &#40;Fig.&#40;b&#41;&#41; consists of a visualization module and Module-I<sup>2</sup> &#40;Fig.&#40;c&#41;&#41; aims intra-mammogram &#40;multi-view&#41; and inter-modal learning.)
![A<sup>2</sup>I<sup>2</sup>](https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/visualization.png)

[comment]: <> (* As illustrated in our workflow, we introduce A<sup>2</sup>I<sup>2</sup>, including abnormality-awareness &#40;Module-A<sup>2</sup>&#41; &#40;Fig.&#40;b&#41;&#41; consists of a visualization module and Module-I<sup>2</sup> &#40;Fig.&#40;c&#41;&#41; aims intra-mammogram &#40;multi-view&#41; and inter-modal learning.)
![A<sup>2</sup>I<sup>2</sup>](https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/word_based_attention.png )


## Usage

Start by [installing PyTorch 1.7.1](https://pytorch.org/get-started/locally/) with the right CUDA version, then clone this repository and install the dependencies.  

```bash
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
$ pip install git@github.com:yawwG/Visualize-what-you-learn.git
$ conda env create -f environment.yml
```

Make sure to download the pretrained weights from [here](https://)(it will be publicly availible soon!) and place it in the `./pretrained` folder.

### Load A<sup>2</sup>I<sup>2</sup> pretrained models 
```python
def load_VSWL(
    name: str = "VSWL_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Load a VSWL model

    Parameters
    ----------
    name : str
        A model name listed by `VSWL2.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model

    Returns
    -------
    VSWL_model : torch.nn.Module
        The VSWL model
    """

    # warnings
    if name in _MODELS:
        ckpt_path = _MODELS[name]
    elif os.path.isfile(name):
        ckpt_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    ckpt = torch.load(ckpt_path)
    # ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("VSWL2.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    VSWL_model = builder.build_VSWL_model(cfg).to(device)
    VSWL_model.load_state_dict(ckpt_dict)

    return VSWL_model
```


### Zeroshot classification, segmentation, localization
```bash
python zeroshot.py
#Check more details from VSML.py including definations of zeroshot applications model.   
```

## Training

This codebase has been developed with python version 3.7, PyTorch version 1.7.1, CUDA 10.2 and pytorch-lightning 1.1.4. 
Example configurations for pretraining and downstream classification can be found in the `./configs`. All training and testing are done using the `run.py` script. For more documentation, please run: 

```bash 
python run.py --help
```

The preprocessing steps for each dataset can be found in `preprocess_datasets.py`

### Representation Learning pretraining

Train the representation learning model with the following command: 

```bash 
python run.py -c pretrain_config.yaml --train
```

### Classification 

Fine-tune the A<sup>2</sup>I<sup>2</sup> pretrained image model for classification with the following command: 

```bash 
python run.py  -c configs/classification_config_1.yaml --train --test --train_pct 1 &
python run.py  -c configs/classification_config_0.1.yaml --train --test --train_pct 0.1 &
python run.py  -c configs/classification_config_0.01.yaml --train --test --train_pct 0.01
```

The **train_pct** flag randomly selects a percentage of the dataset to fine-tune the model. This is use to determine the performance of the model under low data regime.
The dataset using is specified in config.yaml by key("dataset").
### Segmentation/localization

Fine-tune the A<sup>2</sup>I<sup>2</sup> pretrained image model for segmentation/localization with the following command: 

```bash 
python run.py  -c configs/segmentation_config_1.yaml --train --test --train_pct 1 &
python run.py  -c configs/segmentation_config_0.1.yaml --train --test --train_pct 0.1 &
python run.py  -c configs/segmentation_config_0.01.yaml --train --test --train_pct 0.01
```

## Contact details
If you have any questions please contact us. 

Email: ritse.mann@radboudumc.nl (Ritse Mann); taotanjs@gmail.com (Tao Tan); y.gao@nki.nl (Yuan Gao)

Links: [Netherlands Cancer Institute](https://www.nki.nl/), [Radboud University Medical Center](https://www.radboudumc.nl/en/patient-care) and [Maastricht University](https://www.maastrichtuniversity.nl/nl) and [The University of Hong Kong](https://www.hku.hk/) 

<img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/NKI.png" width="166.98" height="87.12"/><img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/RadboudUMC.png" width="231" height="74.58"/><img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/Maastricht.png" width="237.6" height="74.844"/><img src="https://github.com/yawwG/Visualize-what-you-learn/blob/main/figures/hku.png" width="94" height="94.844"/>  

