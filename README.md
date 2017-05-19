# VAE with a VampPrior
This is a PyTorch implementation of a new prior ("Variational Mixture of Posteriors" prior, or VampPrior for short) for the variational auto-encoder framework with one layer and two layers of stochastic hidden units as described in the following paper:
* Jakub M. Tomczak, Max Welling, VAE with a VampPrior, [arXiv preprint](https://arxiv.org/abs/1611.09630), 2017

## Data
The experiments can be run pn four datasets:
* static MNIST: links to the datasets can found at [link](https://github.com/yburda/iwae/tree/master/datasets/BinaryMNIST);
* binary MNIST: the dataset is loaded from [Keras](https://keras.io/);
* OMNIGLOT: the dataset could be downloaded from [link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat);
* Caltech 101 Silhouettes: the dataset could be downloaded from [link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat).

## Run the experiment
1. Set-up your experiment in `experiment.py`.
2. Run experiment:
```bash
python run_experiment.py
```
## Models
You can run a vanilla VAE ('vae'), a one-layered VAE with the VampPrior ('vae_ vampprior') or a two-layered VAE with the VampPrior by setting `model_name` to either `vae`, `vae_vampprior` or `vae_vampprior_2level`, respectively.

## Citation

Please cite our paper if you use this code in your research:

```
@article{TW:2017,
  title={{VAE with a VampPrior}},
  author={Tomczak, Jakub M and Welling, Max},
  journal={arXiv},
  year={2017}
}
```

## Acknowledgments
The research conducted by Jakub M. Tomczak was funded by the European Commission within the Marie Skłodowska-Curie Individual Fellowship (Grant No. 702666, ”Deep learning and Bayesian inference for medical imaging”).
