# fearless-ep

Code to accompany the paper ["Fearless Stochasticity in Expectation Propagation"](https://arxiv.org/abs/2406.01801) (NeurIPS 2024).

To install the package, the the following command inside the repository directory:
```bash
pip install -e .
```

In order to recreate the paper experiments, first create a reference posterior by running EP to (approximate) convergence with a large number of samples. In order to do this in a reasonable amount of time we can run several stages, gradually increasing the number of samples while decreasing the learning rate. Below is an example of how to create a target for the hierarchical logistic regression (with multivariate normal prior) experiment in 2 stages.
```bash
python experiments/hlr_mvn.py --method=EP --n-samp=10_000 --n-outer=100 --lr=.05 --n-warmup=1_000 --warmup-interval=1 --save-checkpoint-path=hlr-mvn-ep-checkpoint.pkl
python experiments/hlr_mvn.py --method=EP --n-samp=100_000 --n-outer=10 --lr=.025 --n-warmup=10_000 --warmup-interval=1 --load-checkpoint-path=hlr-mvn-ep-checkpoint.pkl --save-reference-path=hlr-mvn-target.pkl
```

The reference posterior will be saved to the path specified by `--save-reference-path`, which can then be used in subsequent experiments. The experiments will log KL divergences between the current approximation and the reference. Below is an example using EP-$`\eta`$.
```bash
python experiments/hlr_mvn.py --method=EP_ETA --n-samp=1 --n-outer=100_000 --lr=1e-4 --n-warmup=200 --warmup-interval=400 --load-reference-path=hlr-mvn-target.pkl
```

Note that we have only included synthetic datasets in this repository. In order to recreate most of the experiments in the paper it will be necessary to download the original datasets, details of which can be found in the paper. The exception to this is the cosmic radiation model, for which the synthetic data generation used here is the same as is used in the paper.
