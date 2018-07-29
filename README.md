# Clique-single
NN used to solve MURA dataset, Clique-single allocates a single gpu using CliqueNet. Further tuning is required. 
## Usage
- Dowload and unzip MURA dataset
- Clone this repository
- Put MURA dataset into dataset folder
- run
```python train.py --gpu [gpu id] --k [filters per layer] --T [all layers of three blocks] --dir [path to save models]```
## CliqueNet
```@article{yang18,
 author={Yibo Yang and Zhisheng Zhong and Tiancheng Shen and Zhouchen Lin},
 title={Convolutional Neural Networks with Alternately Updated Clique},
 journal={arXiv preprint arXiv:1802.10419},
 year={2018}}
```
## MURA dataset
```@misc{1712.06957,
Author = {Pranav Rajpurkar and Jeremy Irvin and Aarti Bagul and Daisy Ding and Tony Duan and Hershel Mehta and Brandon Yang and Kaylie Zhu and Dillon Laird and Robyn L. Ball and Curtis Langlotz and Katie Shpanskaya and Matthew P. Lungren and Andrew Ng},
Title = {MURA Dataset: Towards Radiologist-Level Abnormality Detection in Musculoskeletal Radiographs},
Year = {2017},
Eprint = {arXiv:1712.06957},}
```
Thanks [mingfengwan](https://github.com/mingfengwan) for the helping the project and thanks [iboing](https://github.com/iboing) for providing the sample code. 
