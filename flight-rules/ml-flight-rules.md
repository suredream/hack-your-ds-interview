<h1 align='center'>
    Flight Rules for Machine Learning
</h1>

<h4 align='center'>
    A guide for astronauts (now, people doing machine learning) about what to do when things go wrong.
</h4>

<p align='center'>
    <a href="https://www.producthunt.com/posts/machine-learning-flight-rules?utm_source=badge-featured&utm_medium=badge&utm_souce=badge-machine-learning-flight-rules" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=175170&theme=dark" alt="Machine Learning Flight Rules - Tips and tricks for doing machine learning right | Product Hunt Embed" style="width: 250px; height: 54px;" width="250px" height="54px" />
    </a>
    <a href="https://forthebadge.com">
        <img src="https://forthebadge.com/images/badges/built-with-love.svg" alt="forthebadge">
    </a>
    <a href="https://forthebadge.com">
        <img src="https://forthebadge.com/images/badges/cc-sa.svg" alt="forthebadge">
    </a>
    <a href="https://github.com/prettier/prettier">
        <img src="https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square" alt="code style: prettier" />
    </a>
    <a href="http://makeapullrequest.com">
        <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome">
    </a>
    <a href="https://github.com/bkkaggle/machine-learning-flight-rules/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/bkkaggle/machine-learning-flight-rules">
    </a>
</p>

<p align='center'>
    <a href='#what-are-flight-rules'>What are flight rules?</a> •
    <a href='#contributing'>Contributing</a> •
    <a href='#authors'>Authors</a> •
    <a href='#license'>License</a> •
    <a href='#acknowledgements'>Acknowledgements</a>
</p>

<p align='center'><strong>Made by <a href='https://github.com/bkkaggle'>Bilal Khan</a> • https://bilal.software</strong></p>

## What are "flight rules"?

_Copied from: https://github.com/k88hudson/git-flight-rules_

> _Flight Rules_ are the hard-earned body of knowledge recorded in manuals that list, step-by-step, what to do if X occurs, and why. Essentially, they are extremely detailed, scenario-specific standard operating procedures. [...]

> NASA has been capturing our missteps, disasters and solutions since the early 1960s, when Mercury-era ground teams first started gathering "lessons learned" into a compendium that now lists thousands of problematic situations, from engine failure to busted hatch handles to computer glitches, and their solutions.

&mdash; Chris Hadfield, _An Astronaut's Guide to Life_.

## Table Of Contents

<details>
  <summary>Table Of Contents</summary>

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

-   [General tips](#general-tips)
    -   [Look at the wrongly classified predictions of your network](#look-at-the-wrongly-classified-predictions-of-your-network)
    -   [Always set the random seed](#always-set-the-random-seed)
    -   [Make a baseline and then increase the size of your model until it overfits](#make-a-baseline-and-then-increase-the-size-of-your-model-until-it-overfits)
        -   [Use a very simplified baseline to test that your code works correctly](#use-a-very-simplified-baseline-to-test-that-your-code-works-correctly)
        -   [Overfit on a single batch](#overfit-on-a-single-batch)
        -   [Be sure that you're data has been correctly processed](#be-sure-that-youre-data-has-been-correctly-processed)
        -   [Simple models -> complex models](#simple-models---complex-models)
        -   [Start with a simple optimizer](#start-with-a-simple-optimizer)
        -   [Change one thing at a time](#change-one-thing-at-a-time)
    -   [Regularize your model](#regularize-your-model)
        -   [Get more data](#get-more-data)
        -   [Data augmentation](#data-augmentation)
        -   [Use a pretrained network](#use-a-pretrained-network)
        -   [Decrease the batch size](#decrease-the-batch-size)
        -   [Use early stopping](#use-early-stopping)
    -   [Squeeze out more performance out of the network](#squeeze-out-more-performance-out-of-the-network)
        -   [Ensemble](#ensemble)
        -   [Use early stopping on the val metric](#use-early-stopping-on-the-val-metric)
    -   [Learn to deal with long iteration times](#learn-to-deal-with-long-iteration-times)
    -   [Keep a log of what you're working on](#keep-a-log-of-what-youre-working-on)
    -   [Try to predict how your code will fail](#try-to-predict-how-your-code-will-fail)
    -   [Resources](#resources)
-   [Advanced tips](#advanced-tips)
    -   [Basic architectures are sometimes better](#basic-architectures-are-sometimes-better)
    -   [Be sure that code that you copied from Github or Stackoverflow is correct](#be-sure-that-code-that-you-copied-from-github-or-stackoverflow-is-correct)
    -   [Don't excessively tune hyperparameters](#dont-excessively-tune-hyperparameters)
    -   [Set up cyclic learning rates correctly](#set-up-cyclic-learning-rates-correctly)
    -   [Manually init layers](#manually-init-layers)
    -   [Mixed/half precision training](#mixedhalf-precision-training)
        -   [What is the difference between mixed and half precision training?](#what-is-the-difference-between-mixed-and-half-precision-training)
    -   [Apex won't install on GCP's deep learning vm](#apex-wont-install-on-gcps-deep-learning-vm)
        -   [Resources](#resources-1)
    -   [gradient accumulation](#gradient-accumulation)
    -   [multi gpu/machine training](#multi-gpumachine-training)
    -   [determinism](#determinism)
    -   [Initalization](#initalization)
        -   [Types of intialization](#types-of-intialization)
            -   [Xavier or glorot initialization](#xavier-or-glorot-initialization)
            -   [Kaiming or he initialization](#kaiming-or-he-initialization)
        -   [Gain](#gain)
        -   [Pytorch defaults](#pytorch-defaults)
        -   [Resources](#resources-2)
    -   [Normalization](#normalization)
        -   [Batch norm](#batch-norm)
            -   [You can't use a batch size of 1 with batch norm](#you-cant-use-a-batch-size-of-1-with-batch-norm)
            -   [Be sure to use model.eval() with batch norm](#be-sure-to-use-modeleval-with-batch-norm)
            -   [Resources](#resources-3)
-   [Common errors](#common-errors)
-   [Pytorch](#pytorch)
    -   [Losses](#losses)
        -   [cross_entropy vs nll loss for multi-class classification](#cross_entropy-vs-nll-loss-for-multi-class-classification)
        -   [binary_cross_entropy vs binary_cross_entropy_with_logits for binary classification tasks](#binary_cross_entropy-vs-binary_cross_entropy_with_logits-for-binary-classification-tasks)
        -   [Binary classification vs multi-class classification](#binary-classification-vs-multi-class-classification)
        -   [Pin memory in the dataloader](#pin-memory-in-the-dataloader)
        -   [`model.eval()` vs `torch.no_grad()`](#modeleval-vs-torchno_grad)
        -   [What to use for `num_workers` in the dataloader](#what-to-use-for-num_workers-in-the-dataloader)
    -   [Tensorboard](#tensorboard)
        -   [How to use it](#how-to-use-it)
    -   [Use Tensorboard in a kaggle kernel](#use-tensorboard-in-a-kaggle-kernel)
        -   [What do the histograms mean?](#what-do-the-histograms-mean)
    -   [Common errors](#common-errors-1)
        -   [RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation](#runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation)
        -   [Creating MTGP constants failed error](#creating-mtgp-constants-failed-error)
        -   [ValueError: Expected more than 1 value per channel when training](#valueerror-expected-more-than-1-value-per-channel-when-training)
    -   [How to](#how-to)
        -   [How to implement gradient clipping](#how-to-implement-gradient-clipping)
        -   [How to implement global max/avg pooling](#how-to-implement-global-maxavg-pooling)
        -   [How to release gpu memory](#how-to-release-gpu-memory)
        -   [How to concatenate hidden states of a bidirectional lstm](#how-to-concatenate-hidden-states-of-a-bidirectional-lstm)
    -   [Torchtext](#torchtext)
        -   [Sort batches by length](#sort-batches-by-length)
        -   [Pretrained embeddings](#pretrained-embeddings)
        -   [Serializing datasets](#serializing-datasets)
-   [Kaggle](#kaggle)
    -   [Tips](#tips)
        -   [Trust your local validation](#trust-your-local-validation)
        -   [Optimize for the metric](#optimize-for-the-metric)
        -   [Something that works for someone might not work for you](#something-that-works-for-someone-might-not-work-for-you)
    -   [Tricks](#tricks)
        -   [Removing negative samples from a dataset is equivalent to loss weighting](#removing-negative-samples-from-a-dataset-is-equivalent-to-loss-weighting)
        -   [Thresholding](#thresholding)
            -   [Using the optimal threshold on a dataset can lead to brittle results](#using-the-optimal-threshold-on-a-dataset-can-lead-to-brittle-results)
        -   [Shakeup](#shakeup)
    -   [Encoding categorical features](#encoding-categorical-features)
    -   [Optimizing code](#optimizing-code)
        -   [Save processed datasets to disk](#save-processed-datasets-to-disk)
        -   [Use multiprocessing](#use-multiprocessing)
    -   [Data Leaks](#data-leaks)
    -   [Tools](#tools)
        -   [CTR (Click Through Rate prediction) tools](#ctr-click-through-rate-prediction-tools)
        -   [FTRL (Follow The Regularized Leader)](#ftrl-follow-the-regularized-leader)
    -   [Ensembling](#ensembling)
        -   [Correlation](#correlation)
-   [Semantic segmentation](#semantic-segmentation)
-   [NLP](#nlp)
    -   [awd-LSTM](#awd-lstm)
    -   [Multitask learning](#multitask-learning)
    -   [Combine pretrained embeddings](#combine-pretrained-embeddings)
    -   [Reinitialize random embedding matrices between models](#reinitialize-random-embedding-matrices-between-models)
    -   [Try out dropout or gaussian noise after the embedding layer](#try-out-dropout-or-gaussian-noise-after-the-embedding-layer)
    -   [Correctly use masking with softmax](#correctly-use-masking-with-softmax)
    -   [Use dynamic minibatches when training sequence models](#use-dynamic-minibatches-when-training-sequence-models)
    -   [Reduce the amount of OOV (Out Of Vocabulary) words](#reduce-the-amount-of-oov-out-of-vocabulary-words)
    -   [Creating a vocabulary on the train, val sets between folds can lead to information being leaked and artificially increasing your score](#creating-a-vocabulary-on-the-train-val-sets-between-folds-can-lead-to-information-being-leaked-and-artificially-increasing-your-score)
    -   [How to use `pad_packed_sequence` and `pack_padded_sequence`](#how-to-use-pad_packed_sequence-and-pack_padded_sequence)
    -   [Transformers](#transformers)
-   [Gradient boosting](#gradient-boosting)
    -   [How to set hyperparameters](#how-to-set-hyperparameters)
    -   [Resources](#resources-4)
-   [Setting up your environment](#setting-up-your-environment)
    -   [Jupyter notebooks](#jupyter-notebooks)
    -   [Python 3.6+](#python-36)
        -   [Conda](#conda)
-   [Build your own library](#build-your-own-library)
-   [Resources](#resources-5)
    -   [Essential tools](#essential-tools)
    -   [Model zoos](#model-zoos)
    -   [Arxiv alternatives](#arxiv-alternatives)
    -   [Demos](#demos)
    -   [Link aggregators](#link-aggregators)
    -   [Machine learning as a service](#machine-learning-as-a-service)
    -   [Coreml](#coreml)
    -   [Courses](#courses)
    -   [Miscelaneous](#miscelaneous)
-   [Contributing](#contributing)
-   [Authors](#authors)
-   [License](#license)
-   [Acknowledgements](#acknowledgements)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

</details>

## General tips

https://karpathy.github.io/2019/04/25/recipe has some great best practices for training neural networks. Some of his tips include:

### Look at the wrongly classified predictions of your network

This can help tell you what might be wrong with your dataset or model.

### Always set the random seed

This will prevent (most, but not all!) variation in results between otherwise identical training runs.

### Make a baseline and then increase the size of your model until it overfits

#### Use a very simplified baseline to test that your code works correctly

Use a simple model (e.g. a small resnet18 or linear regression) and confirm that your code works properly and as it is supposed to.

#### Overfit on a single batch

Try using as small of a batch size as you can (if you're using batch normalization, that would be a batch of two examples). Your loss should go down to zero within a few iterations. If it doesn't, that means you have a problem somewhere in your code.

#### Be sure that you're data has been correctly processed

Visualize your input data right before the `out = model(x)` to be sure that the data being sent to the network is correct (data has normalized properly, augmentations have been applied correctly, etc)

#### Simple models -> complex models

In most cases, start with a simple model (eg: resnet18) then go on to using larger and more complex models (eg: SE-ResNeXt-101).

#### Start with a simple optimizer

Adam is almost always a safe choice, It works well and doesn't need extensive hyperparameter tuning. Kaparthy suggests using it with a learning rate of 3e-4.
I usually start with SGD with a learning rate of 0.1 and a momentum of 0.9 for most image classification and segmentation tasks.

#### Change one thing at a time

Change one hyperparameter/augmentation/architecture and see its effects on the performance of your network. Changing multiple things at a time won't tell you what changes helped and which didn't.

### Regularize your model

#### Get more data

Training on more data will always decrease the amount of overfitting and is the easiest way to regularize a model

#### Data augmentation

This will artificially increase the size of your dataset and is the next best thing to collecting more data. Be sure that the augmentations you use make sense in the context of the task (flipping images of text in an OCR task left to right will hurt your model instead of helping it).

#### Use a pretrained network

Pretrained networks (usually on Imagenet) help jumpstart your model especially when you have a smaller dataset. The domain of the pretrained network doesn't usually prevent it from helping although pretraining on a similar domain will be better.

#### Decrease the batch size

Smaller batch sizes usually help increase regularization

#### Use early stopping

Use the validation loss to only save the best performing checkpoint of the network after the val loss hasn't gone down for a certain number of epochs

### Squeeze out more performance out of the network

#### Ensemble

Ensemble multiple models either trained on different cross validation splits of the dataset or using different architectures. This always boosts performance by a few percentage points and gives you a more confident measure of the performance of the model on the dataset. Averaging metrics from models in an ensemble will help you figure out whether a change in the model is actually an improvement or random noise.

#### Use early stopping on the val metric

-   Increase the size of the model until you overfit, then add regularization
-   augmentation on mask
-   correlation in ensembles
-   noise in ensembling

---

Another great resource for best practices when training neural networks is (http://amid.fish/reproducing-deep-rl). This article focused on best practices for deep rl, but most of its recommendations are still useful on normal machine learning. Some of these tips include:

### Learn to deal with long iteration times

Most normal programming (web development, IOS development, etc) iteration times usually range in the seconds, but iteration times in machine learning range from minutes to hours. This means that "experimenting a lot and thinking a little", which is usually fine in other programming contexts, will make you waste a lot of time waiting for a training run to finish. Instead, spending more time thinking about what your code does and how it might not work will help you make less mistakes and waste less time.

### Keep a log of what you're working on

Keeping records (tensorboard graphs/model checkpoints/metrics) of training runs and configurations will really help you out when figuring out what worked and what didn't. Additionally, keeping track of what you're working on and your mindset as you're working through a problem will help you when you have to come back to your work days or weeks later.

### Try to predict how your code will fail

Doing this will cut down on the amount of failures that seem obvious in retrospect. I've sometimes had problems where I knew what was wrong with my code before going through the code to debug it. To stop making as many obvious mistakes, I wouldn't start a new training run if I was uncertain about whether it would work, and then would find and fix what might have gone wrong.

### Resources

-   https://karpathy.github.io/2019/04/25/recipe
-   http://amid.fish/reproducing-deep-rl

## Advanced tips

-   some tips should be taken with a grain of salt
-   from: https://gist.github.com/bkkaggle/67bb9b5e6132e5d3c30e366c8d403369

### Basic architectures are sometimes better

Always using the latest, most advanced, SOTA model for a task isn't always the best choice. For example, Although more advanced semantic segmentation models like deeplab and pspnet are SOTA on datasets like PASCAL VOC and cityscapes, simpler architectures like U-nets are easier to train and adapt to new tasks and preform almost just as well on several recent kaggle competitions (https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/107824#latest-623920) (https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69291#latest-592781).

### Be sure that code that you copied from Github or Stackoverflow is correct

It's a good idea to check code from Github and Stackoverflow to make sure it is correct and that you are using it in the correct way. In the Quora insincere questions classification Kaggle competition, a popular implementation of attention summed up the weighted features instead of weighting the actual features with the attention weights (https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/79911) (https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76583#450210).

### Don't excessively tune hyperparameters

Every time you tune hyperparameters on a validation set, you risk overfitting those hyperparameters to that validation set. If done correctly, the improvement from having better hyperparameters will outweigh the risk of having hyperparameters that don't work well on the test set.

### Set up cyclic learning rates correctly

If you're using a cyclic learning rate, be sure that the learning rate is at it's lowest point when you have finished training.

### Manually init layers

Pytorch will automatically initialize layers for you, but depending on your activation function, you might want to use the correct gain for your activation function. Take a look at the pytorch [documentation](https://pytorch.org/docs/stable/nn.init.html) for more information.

### Mixed/half precision training

Mixed or half precision training lets you train on larger batch sizes and can speed up your training. Take a look at [this](https://discuss.pytorch.org/t/training-with-half-precision/11815) if you want to simply use half precision training.

#### What is the difference between mixed and half precision training?

Nvidia's Volta and Turing GPUs contain tensor cores that can do fast fp16 matrix multiplications and significantly speed up your training.

"True" half precision training casts the inputs and the model's parameters to 16 bit floats and computes everything using 16 bit floats. The advantages of this are that fp16 floats only use half the amount of vram as normal fp32 floats, letting you double the batch size while training. This is the fastest and most optimized way to take advantage of tensor cores, but comes at a cost. Using fp16 floats for the model's parameters and batch norm statistics means that if the gradients are small enough, they can underflow and be replaced by zeros.

Mixed precision solves these problems by keeping a master copy of the model's parameters in 32 bit floats. The inputs and the model's parameters are still cast to fp16, but after the backwards pass, the gradients are copied to the master copy and cast to fp32. The parameters are updated in fp32 to prevent gradients from underflowing, and the new, updated master copy's parameters are cast to fp16 and copied to the original fp16 model. Nvidia's apex library recommends using mixed precision in a different way by casting inputs to tensor core-friendly operations to fp16 and keeping other operations in fp32. Both of these mixed precision approaches have an overhead compared to half precision training, but are faster and use less vram than fp32 training.

Take a look at (https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) for more information.

### Apex won't install on GCP's deep learning vm

This is a known issue with apex, take a look at (https://github.com/NVIDIA/apex/issues/259) for some possible solutions.

#### Resources

If you're using pytorch, Nvidia's apex library (https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) is the easiest way to start using mixed precision.
If you want to read more about half and mixed precision training, take a look at https://forums.fast.ai/t/mixed-precision-training/20720

### gradient accumulation

If you want to train larger batches on a gpu without enough vram, gradient accumulation can help you out.

The basic idea is this: call `optimizer.step()` every n minibatches, accumulating the gradients at each minibatch, effectively training on a minibatch of size `batch_size x n`.

Here's a example showing how you could use gradient accumulation in pytorch, from (https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3#file-gradient_accumulation-py):

```python
model.zero_grad()                                   # Reset gradients tensors
for i, (inputs, labels) in enumerate(training_set):
    predictions = model(inputs)                     # Forward pass
    loss = loss_function(predictions, labels)       # Compute loss function
    loss = loss / accumulation_steps                # Normalize our loss (if averaged)
    loss.backward()                                 # Backward pass
    if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
        optimizer.step()                            # Now we can do an optimizer step
        model.zero_grad()                           # Reset gradients tensors
        if (i+1) % evaluation_steps == 0:           # Evaluate the model when we...
            evaluate_model()                        # ...have no gradients accumulated
```

If you want to read more about gradient accumulation, check out this blog post (https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)

### multi gpu/machine training

If you have multiple gpus, you can easily convert your current code to train your model on multiple gpus. Just follow the official tutorials on pytorch.org (https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html). The only problem with this is that Pytorch's build in `DataParallel` will gather the outputs from all the other gpus to gpu 1 to compute the loss and calculate gradients, using up more vram. There _is_ an alternative to this though, just use this alternative balanced data parallel implementation (https://gist.github.com/thomwolf/7e2407fbd5945f07821adae3d9fd1312).

Take a look at (https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255#) for more information about multi gpu and distributed training.

### determinism

Pytorch will give you different results every time you run a script unless you set random seeds for python, numpy, and pytorch. Fortunately, doing this is very simple and only requires you to add a few lines to the top of each python file. There is a catch though, setting `torch.backends.cudnn.deterministic` to `True` will slightly slow down your network.

```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```

If you want a simple one-line way to do this, check out my `pytorch_zoo` library on github (https://github.com/bkkaggle/pytorch_zoo#seed_environmentseed42).

```python
from pytorch_zoo.utils import seed_environment

seed_environment(42)
```

If you want more information on determinism in pytorch, take a look at these links:

-   https://discuss.pytorch.org/t/how-to-get-deterministic-behavior/18177/7
-   https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/72770
-   https://www.kaggle.com/bminixhofer/deterministic-neural-networks-using-pytorch
-   https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/72040

### Initalization

I made some notes about initialization in [this](https://gist.github.com/bkkaggle/58d4e58ac2a5101e42e2d1af9399c638) gist.

The idea behind initializing weights in specific ways instead of by random is to keep the means and stddevs of the activations close to 0 and 1 respectively, preventing activations and therefore the gradients from exploding or vanishing.

#### Types of intialization

##### Xavier or glorot initialization

-   uniform initialization
    -   bounds a uniform distribution between +/- sqrt(6 / (c_in + c_out))
-   normal initialization
    -   multiplys a normal distribution with mean 0 and stddev 1 by sqrt(2 / (c_in + c_out))
    -   Another way to do this is create a normal distribution with mean 0 and stddev sqrt(2 / (c_in + c_out))
-   The logic behind this is to try to keep identical variances across layers

##### Kaiming or he initialization

-   when using a relu activation, stddevs will be close to sqrt(c_in)/sqrt(2), so multiplying the normally distributed activations by sqrt(2/c_in) will make the activations have a stddev close to 1

-   uniform initialization
    -   bound a uniform distribution between +/- sqrt(6 / c_in)
-   normal initialization
    -   multiply a normal distribution by sqrt(2 / c_in)
    -   or create a normal distribution with mean 0 and stddev sqrt(2 / c_in)

#### Gain

-   multiplied to init bounds/stddevs
-   sqrt(2) for relu
-   none for kaiming

#### Pytorch defaults

-   most layers are initialized with kaiming uniform as a reasonable default
-   use kaiming with correct gain (https://pytorch.org/docs/stable/nn.html#torch.nn.init.calculate_gain)

#### Resources

-   https://github.com/pytorch/pytorch/issues/15314
-   https://medium.com/@sakeshpusuluri123/activation-functions-and-weight-initialization-in-deep-learning-ebc326e62a5c
-   https://pytorch.org/docs/stable/_modules/torch/nn/init.html
-   https://discuss.pytorch.org/t/whats-the-default-initialization-methods-for-layers/3157/21
-   https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
-   https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
-   https://pytorch.org/docs/stable/nn.html#torch.nn.init.calculate_gain
-   https://github.com/mratsim/Arraymancer/blob/master/src/nn/init.nim
-   https://jamesmccaffrey.wordpress.com/2018/08/21/pytorch-neural-network-weights-and-biases-initialization/

### Normalization

#### Batch norm

The original batch normalization paper put the batch norm layer before the activation function, recent research shows that putting the batch norm layer after the activation gives better results. A great article on batch norm and why it works can be found here (https://blog.paperspace.com/busting-the-myths-about-batch-normalization/).

##### You can't use a batch size of 1 with batch norm

Batch norm relies on the mean and variance of all the elements in a batch, it won't work if you're using a batch size of one while training, so either skip over any leftover batches with batch sizes of 1 or increase the batch size to atleast 2.

##### Be sure to use model.eval() with batch norm

Run `model.eval()` before your validation loop to make sure pytorch uses the running mean and variance calculated over the training set. Also make sure to call `model.train()` before your training loop to start calculating the batch norm statistics again. You can read more about this at (https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146)

##### Resources

http://mlexplained.com/2018/11/30/an-overview-of-normalization-methods-in-deep-learning/ is a really good blog post on the different types of normalizations and when to them.

## Common errors

## Pytorch

### Losses

#### cross_entropy vs nll loss for multi-class classification

Either pass the logits for a multi-class classification task to `log_softmax` first, then through the `nll` loss or pass the logits directly to `cross_entropy`. They will give you the same result, but `cross_entropy` is more numerically stable. Use `softmax` separately to convert logits into probabilities for prediction or for calculating metrics. Take a look at (https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html) for more information.

#### binary_cross_entropy vs binary_cross_entropy_with_logits for binary classification tasks

Either pass the logits for a binary classification task to `sigmoid` first, then through `binary_cross_entropy` or pass the logits directly to `binary_cross_entropy_with_logits`. Just as the example above, they will give you the same result, but `binary_cross_entropy` is more numerically stable. Use `sigmoid` separately to conver the logits into probabilities for prediction or for calculating metrics. Again, take a look at (https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html) for more information.

#### Binary classification vs multi-class classification

A binary classification task can also be represented as a multi-class classification task with two classes, positive and negative. They will give you the same result and should be numerically identical.

Here's an example, taken from (https://sebastianraschka.com/faq/docs/pytorch-crossentropy.html), on how you could do this:

```python
>>> import torch
>>> labels = torch.tensor([1, 0, 1], dtype=torch.float)
>>> probas = torch.tensor([0.9, 0.1, 0.8], dtype=torch.float)
>>> torch.nn.functional.binary_cross_entropy(probas, labels)
tensor(0.1446)

>>> labels = torch.tensor([1, 0, 1], dtype=torch.long)
>>> probas = torch.tensor([[0.1, 0.9],
...                        [0.9, 0.1],
...                        [0.2, 0.8]], dtype=torch.float)
>>> torch.nn.functional.nll_loss(torch.log(probas), labels)
tensor(0.1446)
```

#### Pin memory in the dataloader

Set `pin_memory` to `true` in your dataloader to speed up transferring your data from cpu to gpu. Take a look at this for more information (https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723).

#### `model.eval()` vs `torch.no_grad()`

`model.eval()` will switch your dropout and batch norm layers to eval mode, turning off dropout and using the running mean and stddev for the batch norm layers. `torch.no_grad()` will tell pytorch to stop tracking operations, reducing memory usage and speeding up your evaluation loop. To use these properly, run `model.train()` before each training loop, run `model.eval()` before each evaluation loop, and wrap your evaluation loop with `with torch.no_grad():` Take a look at this for more information (https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/11).

#### What to use for `num_workers` in the dataloader

If your gpu utilization fluctuates a lot and generally remains low (< 90%), this might mean that your gpu is waiting for the cpu to finish processing all the elements in your batch and that `num_workers` might be your main bottleneck. `num_workers` in the dataloader is used to tell pytorch how many parallel workers to use to preprocess the data ahead of time. Set `num_workers` to the number of cores that you have in your cpu. This will fully utilize all your cpu cores to minimize the amount of time the gpu spends waiting for the cpu to process the data. If your gpu utilization still remains low, you should get more cpu cores or preprocess the data ahead of time and save it to disk. Take a look at these articles for more information: (https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader) and (https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel).

### Tensorboard

Tensorboard is really useful when you want to view your model's training progress in real time. Now that Pytorch 1.1 is out, you can now log metrics directly to tensorboard from Pytorch.

#### How to use it

Follow these instructions for a quickstart (https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html).

### Use Tensorboard in a kaggle kernel

Just copy this code snippet into a cell at the top of your kernel

```python
!mkdir logs
get_ipython().system_raw('tensorboard --logdir ./logs --host 0.0.0.0 --port 6006 &')
!ssh -o "StrictHostKeyChecking no" -R 80:localhost:6006 serveo.net
```

I also have another quickstart at my [pytorch_zoo](https://github.com/bkkaggle/pytorch_zoo#viewing-training-progress-with-tensorboard-in-a-kaggle-kernel) repository.

#### What do all the Tensorboard histograms mean?

Take a look at these stackoberflow posts:

-   https://stackoverflow.com/questions/42315202/understanding-tensorboard-weight-histograms
-   https://stackoverflow.com/questions/38149622/what-is-a-good-explanation-of-how-to-read-the-histogram-feature-of-tensorboard

### Common errors

#### RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation

In place operations and operations on slices of tensors can cause problems with Pytorch's autograd. To fix this, convert your inplace operation, `x[:, 0, :] += 1`, to a non inplace operation, `x[:, 0, :] = x[:, 0, :].clone() + 1`, and use `.clone()` to avoid problems with operations on tensor slices. Take a look at (https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836) for more information.

#### Creating MTGP constants failed error

This error happens when "using an embedding layer and passing out of range indexes (indexes > num_embeddings)" from (https://discuss.pytorch.org/t/solved-creating-mtgp-constants-failed-error/15084/4). For more information, take a look at (https://discuss.pytorch.org/t/solved-creating-mtgp-constants-failed-error/15084).

#### ValueError: Expected more than 1 value per channel when training

This error happens when you're using a batch size of 1 while training with batch norm. Batch norm expects to have a batch size of at least 2. For more information, take a look at (https://github.com/pytorch/pytorch/issues/4534)

### How to

#### How to implement gradient clipping

Here's the code for gradient clipping:

```python
torch.nn.utils.clip_grad_norm(model.parameters(), value)
```

If you want to read more about gradient clipping in pytorch, take a look at (https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191).

#### How to implement global max/avg pooling

Follow the instructions from (https://discuss.pytorch.org/t/global-max-pooling/1345/2)

#### How to release gpu memory

There is no simple way to do this, but you can release as much memory as you can by running `torch.cuda.empty_cache()`. Take a look at (https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530) for more information.

#### How to concatenate hidden states of a bidirectional lstm

Follow the instructions from (https://discuss.pytorch.org/t/concatenation-of-the-hidden-states-produced-by-a-bidirectional-lstm/3686/2).

### Torchtext

Torchtext is Pytorch's official NLP library, The library's official [docs](https://torchtext.readthedocs.io/en/latest/index.html) are the best way to get started with the library, but are a bit limited and there are some blog posts that help you get a better sense of how to use the library:

-   https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html
-   https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
-   https://pytorch.org/tutorials/beginner/transformer_tutorial.html
-   http://anie.me/On-Torchtext/
-   http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
-   http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/
-   https://towardsdatascience.com/use-torchtext-to-load-nlp-datasets-part-i-5da6f1c89d84
-   https://towardsdatascience.com/use-torchtext-to-load-nlp-datasets-part-ii-f146c8b9a496
-   https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html

#### Sort batches by length

Your recurrent models will train best if all the examples in a batch have similar lengths. Since all the examples in a batch are padded with zeros to the length of the longest example, grouping examples with identical or similar lengths will make your model more efficient and waste less of the GPU's memory. Use the iterator's `sort_key` attribute to tell it to group examples of similar lengths into each batch. If you're using `pack_padded_sequence`, set `sort_within_batch` to `True` since `pack_padded_sequence` expects examples in a batch to be in ascending order. Take a look at [this](https://github.com/pytorch/text/issues/303) for more information.

-   https://github.com/pytorch/text/issues/303

#### Pretrained embeddings

If you want to use a pretrained embedding like word2vec or glove, you will have to load in the pretrained vectors and update the field's vectors.

```
# Load in the vectors
vectors = torchtext.vocab.Vectors('/path/to/vectors')

# Create the text field
text_field = data.Field(tokenize=tokenizer, lower=True, batch_first=True, include_lengths=True)

# Built the vocab for the field using the train dataset
text_field.build_vocab(train_dataset)

# Set the vectors of the field to be the pretrained vectors
text_field.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
```

Take a look at [this](https://discuss.pytorch.org/t/aligning-torchtext-vocab-index-to-loaded-embedding-pre-trained-weights/20878) for more information.

#### Serializing datasets

If you're working with large datasets that take time to load and process, being able to serialize and save processed datasets to disk is a really nice feature. Unfortunately, this feature is [still](https://github.com/pytorch/text/issues/140) a work in progress (the issue was created in 2017, and there doesn't seem to be that much work being done on torchtext as of late 2019), so the only way to do this at the moment is to follow [this](https://towardsdatascience.com/use-torchtext-to-load-nlp-datasets-part-ii-f146c8b9a496) article.

## Kaggle

Here are some of tips and tricks I picked up while participating in kaggle competitions.

### Tips

#### Trust your local validation

Your score on your local validation set should be the most important, and sometimes the only, metric to pay attention to. Creating a validation set that you can trust to tell you whether you are or are not making progress is very important.

#### Optimize for the metric

The goal of kaggle competitions is to get the highest (or lowest, depending on the metric) score on a specific metric. To do this, you might need to modify your model's loss function. For example, if the competition metric penalizes mistakes on rare classes more than common classes, oversampling or weighting the loss in favor of those classes can force the model to optimize for that metric.

#### Something that works for someone might not work for you

Just because someone says on the discussion forum that a particular technique or module works better for them doesn't automatically mean that it will work for you.

### Tricks

#### Removing negative samples from a dataset is equivalent to loss weighting

This usually works well and is easier to do than loss weighting.

#### Thresholding

##### Using the optimal threshold on a dataset can lead to brittle results

If you choose thresholds for (binary) classification problems by choosing whatever value gives you the optimal score on a validation set, the threshold might be overfitting to the specific train-val split or to the specific architecture/hyperparameters. This can have two effects. First, the optimial threshold you found on the val set might not be the optimal threshold on the held out test set, decreasing your score. Second, this makes comparing results between runs with different model architectures or hyperparameters more difficult. Using different thresholds means that a model that is actually worse might get a higher score than a better model if you find a 'lucky' threshold.

#### Shakeup

Shakeup prediction is a powerful tool to predict the likely range of scores for your model when evaluated on an unknown test set. It was first introduced by the winner of a kaggle competition as a way to stabilize his models in (https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36809). It has also been used [here](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/67090) and [here](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/75821).

### Encoding categorical features

Encoding categorical features is a pretty important thing to do when working with tabular data.

Some resources I found for this are:

-   https://www.kaggle.com/c/microsoft-malware-prediction/discussion/79045
-   https://www.kaggle.com/vprokopev/mean-likelihood-encodings-a-comprehensive-study
-   https://www.kaggle.com/tnarik/likelihood-encoding-of-categorical-features
-   https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76668

### Optimizing code

#### Save processed datasets to disk

As long as your dataset isn't too large, saving the processed dataset to disk as a `.pkl` file, then loading it in whenever you need to use it, will save you time and will help increase your GPU utilization.

#### Use multiprocessing

Python's [`multiprocessing`](https://docs.python.org/2/library/multiprocessing.html) library can help you take full advantage of all the cores in your CPU.

### Data Leaks

Finding leaks in a dataset is a difficult, but sometimes useful skill.

Some good examples of how kagglers found leaks are:

-   https://www.kaggle.com/raddar/towards-de-anonymizing-the-data-some-insights
-   https://www.kaggle.com/cpmpml/raddar-magic-explained-a-bit/

### Tools

-   https://github.com/mxbi/mlcrate
-   https://github.com/bkkaggle/pytorch_zoo (I made this)

#### CTR (Click Through Rate prediction) tools

-   https://github.com/guoday/ctrNet-tool
-   https://www.kaggle.com/c/avazu-ctr-prediction/discussion/10927
-   https://www.kaggle.com/c/microsoft-malware-prediction/discussion/75149
-   https://www.kaggle.com/scirpus/microsoft-libffm-munger
-   https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56497#331685

#### FTRL (Follow The Regularized Leader)

-   https://www.kaggle.com/c/microsoft-malware-prediction/discussion/75246

### Ensembling

#### Correlation

Ensembling models with low correlations is better than ensembling models with high correlations.

More information can be found here:

-   https://www.kaggle.com/c/microsoft-malware-prediction/discussion/80368
-   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/51058

## Semantic segmentation

Some good resources for semantic segmentation include:

-   http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review
-   https://tuatini.me/practical-image-segmentation-with-unet/
-   https://www.jeremyjordan.me/semantic-segmentation/#loss
-   https://becominghuman.ai/investigating-focal-and-dice-loss-for-the-kaggle-2018-data-science-bowl-65fb9af4f36c

## NLP

Take a look at some of these blog posts:

-   http://ruder.io/a-review-of-the-recent-history-of-nlp/
-   https://medium.com/huggingface/learning-meaning-in-natural-language-processing-the-semantics-mega-thread-9c0332dfe28e
-   https://medium.com/huggingface/100-times-faster-natural-language-processing-in-python-ee32033bdced

### awd-LSTM

Take a look at these links:

-   https://github.com/salesforce/awd-lstm-lm
-   https://www.fast.ai/2017/08/25/language-modeling-sota/

### Multitask learning

Take a look at these links:

-   http://ruder.io/multi-task/
-   http://ruder.io/multi-task-learning-nlp/

### Combine pretrained embeddings

Adding/concatenating/(weighted) averaging multiple pretrained embeddings almost always leads to a boost in accuracy.

### Reinitialize random embedding matrices between models

Initializing embeddings for unknown words randomly helps increase the diversity between models.

From: (https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/79720)

### Try out dropout or gaussian noise after the embedding layer

It can help increase model diversity and decrease overfitting

### Correctly use masking with softmax

### Use dynamic minibatches when training sequence models

Using this will try to create batches of examples with equal lengths to minimize unncessary padding and wasted calculations. The code to use this is available at (https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779)

### Reduce the amount of OOV (Out Of Vocabulary) words

### Creating a vocabulary on the train, val sets between folds can lead to information being leaked and artificially increasing your score

-   https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/79556

### How to use `pad_packed_sequence` and `pack_padded_sequence`

Take a look at these links:

-   https://discuss.pytorch.org/t/packedsequence-for-seq2seq-model/3907
-   https://discuss.pytorch.org/t/solved-multiple-packedsequence-input-ordering/2106/7

### Transformers

Take a look at these links:

-   https://blog.floydhub.com/the-transformer-in-pytorch/
-   http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/
-   https://jalammar.github.io/illustrated-transformer/

## Gradient boosting

### How to set hyperparameters

Laurae's [website](https://sites.google.com/view/lauraepp/parameters) is the best place to understand what parameters to use and what values to set them to.

### Resources

-   https://www.kaggle.com/c/microsoft-malware-prediction/discussion/78253
-   http://mlexplained.com/2018/01/05/lightgbm-and-xgboost-explained

-   documentation
    -   https://xgboost.readthedocs.io/en/latest/tutorials/model.html
    -   https://lightgbm.readthedocs.io/en/latest/
    -   https://xlearn-doc.readthedocs.io/en/latest/index.html
    -   https://catboost.ai/docs/

## Setting up your environment

### Jupyter notebooks

-   https://stackoverflow.com/questions/43759610/how-to-add-python-3-6-kernel-alongside-3-5-on-jupyter
-   https://forums.fast.ai/t/jupyter-notebook-keyerror-allow-remote-access/24392

### Python 3.6+

-   https://www.rosehosting.com/blog/how-to-install-python-3-6-4-on-debian-9/

#### Conda

-   https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment
-   https://stackoverflow.com/questions/35245401/combining-conda-environment-yml-with-pip-requirements-txt
-   https://stackoverflow.com/questions/42352841/how-to-update-an-existing-conda-environment-with-a-yml-file

## Build your own library

I recently built my own machine learning [library](https://github.com/bkkaggle/L2), here are some of the resources I used:

-   https://medium.com/@florian.caesar/how-to-create-a-machine-learning-framework-from-scratch-in-491-steps-93428369a4eb
-   https://github.com/joelgrus/joelnet
-   https://medium.com/@johan.mabille/how-we-wrote-xtensor-1-n-n-dimensional-containers-f79f9f4966a7
-   https://mlfromscratch.com
-   https://eisenjulian.github.io/deep-learning-in-100-lines/
-   http://blog.ezyang.com/2019/05/pytorch-internals/

## Resources

### Essential tools

-   https://paperswithcode.com - This website lists available implementations of papers along with leaderboards showing which models are currently SOTA on a range of tasks and datasets
-   https://www.arxiv-vanity.com - This site converts PDF papers from Arxiv to mobile-friendly responsive web pages.
-   http://www.arxiv-sanity.com - This site is a better way to keep up to date with popular and interesting papers.

### Model zoos

-   https://modelzoo.co/blog
-   https://modeldepot.io/search
-   https://github.com/sebastianruder/NLP-progress

### Arxiv alternatives

-   https://www.arxiv-vanity.com
-   http://www.arxiv-sanity.com
-   https://www.scihive.org

### Machine learning demos

-   https://ganbreeder.app
-   https://talktotransformer.com
-   https://transformer.huggingface.co
-   https://www.nvidia.com/en-us/research/ai-playground/
-   https://alantian.net/ganshowcase/
-   https://rowanzellers.com/grover/
-   http://nvidia-research-mingyuliu.com/gaugan/
-   http://nvidia-research-mingyuliu.com/petswap/

### Link aggregators

-   https://news.ycombinator.com
-   https://www.sciencewiki.com
-   https://git.news/?ref=producthunt

### Machine learning as a service

-   https://runwayml.com
-   https://supervise.ly

### Coreml

-   https://developer.apple.com/machine-learning/models/
-   https://github.com/huggingface/swift-coreml-transformers
-   https://www.fritz.ai

### Courses

-   https://fast.ai
-   https://www.coursera.org/learn/competitive-data-science
-   https://www.deeplearning.ai
-   https://www.kaggle.com/learn/overview

### Miscelaneous

-   https://markus-beuckelmann.de/blog/boosting-numpy-blas.html
-   https://github.com/Wookai/paper-tips-and-tricks
-   https://github.com/dennybritz/deeplearning-papernotes
-   https://github.com/HarisIqbal88/PlotNeuralNet

# Contributing

I've tried to make sure that all the information in this repository is accurate, but if you find something that you think is wrong, please let me know by opening an issue.

This repository is still a work in progress, so if you find a bug, think there is something missing, or have any suggestions for new features, feel free to open an issue or a pull request. Feel free to use the library or code from it in your own projects, and if you feel that some code used in this project hasn't been properly accredited, please open an issue.

# Authors

-   _Bilal Khan_

# License

This project is licensed under the CC-BY-SA-4.0 License - see the [license](LICENSE) file for details

# Acknowledgements

-   _k88hudson_ - _Parts of https://github.com/k88hudson/git-flight-rules were used in this repository_

This repository was inspired by https://github.com/k88hudson/git-flight-rules and copied over parts of it