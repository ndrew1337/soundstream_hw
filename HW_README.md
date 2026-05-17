# Homework 4 (Neural Audio Codec)

## Task

In this homework, you need to implement [SoundStream](https://arxiv.org/abs/2107.03312) neural audio codec. Steps:

1. Read the paper.
2. Find all the details regarding the method and the experimental setup.
3. Write a training pipeline and the code for the model.
4. Train the model.
5. Evaluate the model and write a report.

We will focus on Speech data only. For the training dataset, use `train-clean-100` partition of [LibriSpeech](https://www.openslr.org/12) (the [Kaggle version](https://www.kaggle.com/datasets/a24998667/librispeech)). Evaluate on `test-clean` of LibriSpeech.

**You are not allowed to use and look at any implementations from the web**. The penalty for doing this will be severe. The details provided in the paper are enough. For some of its modules, e.g. Discriminator, you will need to look at the cited papers to find the architecture details and hyperparameters. **You must write the RVQ module yourself, you cannot use a library version.**

To simplify the task, we provide some additional details/hints.

1. Discriminator uses `LeakyReLU` with 0.2 slope.
2. We do not care about denoising here, so do not add the FiLM module used in the original paper.
3. We only focus on a single bitrate, so do not add bitrate dropout.
4. Train on audio crops, evaluate on full audio.
5. Add commitment loss (MSE between encoder output and its quantized detached version) to avoid encoder outputs drifting from the codebook. Use 1.0 weight for the loss.
6. We use sample rate 16kHz, so we need to adjust the strides to preserve the bitrate around 6k. Use [2, 4, 5, 5].

Also, the SoundStream training setup might not be clear from the paper. Pretend like there was the following paragraph in the paper:

> For the training, we follow the [SEANet](https://arxiv.org/abs/2009.02095) optimization setup with a constant learning rate. The SoundStream is trained for 45000 steps on 0.5s random crops of 16kHz audio with batch size 12 on Kaggle T4 GPU. If the audio is shorter than 0.5s, we pad it with replication.

**Note** using 1s or 2s crops also works but may require more time/GPU memory. You can experiment.

## Code requirements

Your submission **is the GitHub repository**; the code **is written in the project-style, not `ipynb` notebooks**.

For the code you have two options:

1. (Default) -- project-style code, i.e., separate folders/files for different parts of the pipeline (e.g. for the model, data, and training). Code running is done via scripts, e.g. `train.py`. Feel free to design the style yourself and add extra features, such as configuration, if you want.
2. (**Bonus** `+1.0` to your grade) -- use advanced [Project Template](https://github.com/Blinorot/pytorch_project_template) and modify it to train your codec. Run the code via scripts. **Note:** you will be required to use it for the next homework, so getting familiar with it earlier might help you. Feel free to choose any of the code branches as a starting point (maybe you will find the `main` branch easier than the `ASR` one).

**Hint**: to run project style code in Colab\Kaggle, you can do the following:

```bash
!git clone https://YOUR_PRIVATE_TOKEN@github.com/USERNAME/REPO_ID

# change the root dir
%cd REPO_ID

# install packages
...

# run the code
python3 train.py ...
```

Code configuration and `%%writefile` will help you to avoid unnecessary commits.

> [!IMPORTANT]
> No matter which code option you choose, you must write clean code as in the previous homeworks. You will be penalized for unnecessary copy-paste and unreadable code.

Besides, you need to provide demo (see [Demo](#demo)). Hence, you need to save your final model checkpoint and provide it with your submission. You can use HuggingFace Models or Google Drive.

## Other requirements

We do not accept the homework if any of the following requirements are not satisfied:

- The code shall be stored in a public github (or gitlab) repository. (Before the deadline, use a private repo. Make it public after the deadline.)
- All the necessary packages shall be mentioned in `./requirements.txt` or an analog (e.g. it can be `.lock` if you use `uv`).
- All necessary resources (e.g. model checkpoints) shall be downloadable with a script. Mention the script (or lines of code) in the `README.md`.
- `README.md` must clearly describe what the repo is about, how to install it and run the model training/inference.
- You must use `W&B`/`Comet ML` for logging losses, objects (like audio), and performance metrics.
- You shall create a demo notebook (see [Demo](#demo) section).
- You must provide the logs for the training of your final model from the start of the training. Use W&B (CometML) Reports feature.
- Attach a detailed report that includes:
  - Description and result of each experiment.
  - How to reproduce your model?
  - Attach training logs showing the rate of convergence and generated audio. Attach metrics curves (**Note**: STOI is fast, you can calculate it always, NISQA is slower, you may want to calculate it only occasionally, but for the final model **you must provide full final metrics on full test data**.)
  - What worked and what didn't work?
  - What were the major challenges?
  - Special tasks from the [Report](#report) section.

> [!NOTE]
> For training such a complicated model, it is useful to not only log the joint loss, but also the individual terms to monitor if something is going wrong. **You must provide these individual logs**. For codebook, you can also calculate the perplexity of codebook usage. $\text{Perplexity} = \exp\left(-\sum_{i=1}^{K} p_i \log p_i \right)$, where $p_i$ is the is the empirical probability of selecting codebook entry $i$.

> [!TIPS]
> One of the homework goals is to get practice in proper project development. Thus, we will look at your Git history, `README`, `requirements.txt`, etc. Provide a comprehensive `README`, do not indicate packages that are not actually needed in the requirements, write meaningful commit names, etc. Do not hesitate to use `pre-commit` for code formatting (template version includes `black` and `isort`).

## Demo

In addition to providing detailed instructions in the `README`, a great repository shows a demo of the project, i.e., showcases how to use it. This allows end-user to see directly how to apply your code for their needs. The basic version of demo is the inference notebook. You must include such an `ipynb` notebook in your repository.

The structure of the demo:

1. Clones your repository and follows all the installation steps from your `README`.
2. Downloads all the required checkpoints.
3. Given a custom url (provided by a user) to an audio file (for example, this [LJ Speech sample](https://keithito.com/LJ-Speech-Dataset/LJ025-0076.wav)), downloads it and passes through your pretrained codec. Then, it displays the resulting audio.
4. All of this is accompanied with comments explaining what your cell is doing and/or gives some instructions.

> [!IMPORTANT]
> Be sure to check your demo yourself. If the demo is absent or not working, you will get $0$ for the homework. We will use your demo notebook with our audio links to evaluate your submission. **A user only have to pass the link and run the cells to get re-synthesized samples, no other steps shall be expected from the user.**

To ensure fair evaluation, use Google Colab as the testing server for the verification of your demo. If your demo code works in a fresh Colab session, then, there shall not be any problems. If it doesn't work in Colab, we will penalize it.

## Report

To supplement your report, conduct the following analysis.

**Qualitative analysis, in-domain**:

1. Take several real utterances from `test-clean` LibriSpeech dataset.
2. Generate re-synthesized versions using your codec.
3. Compare the generated samples with the corresponding original ones. Do this in the time and time-frequency domains (you can also try STFT instead of MEL or other representations). What differences do you see? Can you understand that the audio is synthesized by listening to them? Can you do it by looking at the waveform or spectrogram? Explain the results and come up with some conclusions.

**External Dataset Analysis**:

1. Re-do the Qualitative analysis on samples from another English dataset (not LibriSpeech). For example on some audio that is not that clean. Analyze the differences between real and generated samples. Also, say how the codec quality changed on another dataset.

2. Re-do it on samples with Russian speech. Does the code performance differ? Why yes or why not? Is it expected? In general, how codec should perform on another languages and how to get the best performance? What about low-resource languages without data to train on?

**Quantitative analysis**

Supplement your analysis with some statistics on a big-enough set of data. Think about statistics that might be different on real and generated data. Describe the statistics, provide their values, analyze the differences.

> [!NOTE]
> 1-word and 1-sentence analyses will be penalized. This is a research task.

You have 2 options for the report submission format.

- Main report (logs, etc. from the [requirements](#other-requirements) section) -- in Comet ML/W&B Report. Analysis -- in a separate `ipynb` file (include it in the repo).
- Everything in one Comet ML/W&B Report.

If you choose the first option, your analysis notebook must only contain `markdown` cells with descriptions and imports that will produce images/tables. That is, your code cells must be designed in the following way:

```python
# names here are an example
from analysis import get_in_domain_images

get_in_domain_images() # plots the images
```

## Grade

```
grade = Codec_Performance + 0.5 * (quality of code and report)
```

`Codec_Performance` is based on two metrics, [STOI](https://lightning.ai/docs/torchmetrics/stable/audio/short_time_objective_intelligibility.html) and [NISQA v2.0](https://lightning.ai/docs/torchmetrics/stable/audio/non_intrusive_speech_quality_assessment.html).

| Grade | STOI   | NISQA  |
| ----: | ------ | ------ |
|     5 | > 0.80 | > 2.25 |
|     4 | > 0.78 | > 2.00 |
|     3 | > 0.75 | > 1.50 |
|     2 | > 0.72 | > 1.00 |
|     1 | > 0.60 | > 0.75 |

**You need to pass both metrics to get the grade.**

## Bonuses

- Pytorch Template bonus from the [Code requirements](#code-requirements).

- `0.5 points`. Compare your codec trained with adversarial (GAN) losses (original setup) with the codec trained without them. Explain and show the differences. **Note**: Codec performance grade is based on the adversarial model, not this one.

- `3 points`. Train and evaluate a codec-based Text-To-Speech system on a [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/). Since this task requires a high-quality codec, take a pre-trained codec, [X-Codec](https://huggingface.co/hf-audio/xcodec-hubert-librispeech). Instead of predicting all residual codes, predict only the first one and drop the rest. Also take a pre-trained Language Model, [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) as a backbone. To obtain a TTS system, you need to replace output vocabulary with Codec codes and predict codes corresponding to an audio given a text transcription. You can try training only an LM head or also do LoRA. Add a new section to the report showing your experiments with your comments, STOI/NISQA scores, and text-audio samples. Add a demo for this bonus with a possibility of providing a custom prompt.
