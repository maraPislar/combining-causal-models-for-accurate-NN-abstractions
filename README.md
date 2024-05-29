# Disentangling Causal Models with DAS
Master Thesis - Discovering Causal Models which Align with LLMs Using Distributed Alignment Search (DAS)

The code of this thesis was based on the one provided by the authors in their [pyvene repo](https://github.com/stanfordnlp/pyvene).

## DAS on MLP

DAS can be applied on any type of neural network. It offers an intuition on what causal mechanisms better describe the relationships between neural network inputs.

### Reproducing DAS results

The authors of DAS provide tutorials on how their library works. [The tutorial]((https://github.com/stanfordnlp/pyvene/blob/main/tutorials/advanced_tutorials/DAS_Main_Introduction.ipynb)) on how DAS works when aligning a causal model with an MLP was the main source of inspiration for the code of this thesis. The MLP was trained on a hierarchical equality task ([Premack 1983](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/7DF6F2D22838F7546AF7279679F3571D/S0140525X00015077a.pdf/div-class-title-the-codes-of-man-and-beasts-div.pdf)). The input is two pairs of objects and the output is **True** if both pairs contain the same object or if both pairs contain different objects and **False** otherwise.  For example, `AABB` and `ABCD` are both labeled **True**, while `ABCC` and `BBCD` are both labeled **False**. To reproduce their results, one can simply run:

```
python3 reproduce_das_experiment.py
```

### Experimenting with another task

To further challenge their hypothesis and check that their claims hold, we experiment with a different task. The pattern `ABAB` is the one which would yield `True` in this case. Namely, the first and third inputs should be the same, same for the third and forth. The notebook can be found [here](https://github.com/maraPislar/align-transformers/blob/main/tutorials/advanced_tutorials/DAS_Pattern_Matching_Task.ipynb), or one can run:

```
python3 mlp/pattern_matching_das.py
```

### Ablation study: wrong causal model?

What if DAS is trained on the counterfactual data generated from a causal model which models `ABAB`, but is then tested on data generated from causal models modelling `AABB` or `ABBA`. Intuitively, the IIA yielded when using data from `ABAB` should be much higher than when testing on the other datasets. This can also be seen as sanity checking DAS if it actually learns to align a specific causal model to a neural network.

To run this experiment, one can run:

```
python3 mlp/wrong_causal_graph_ablation_study.py
```

## DAS on LLMs

DAS can even be applied to LLMs. Imagine having a simple causal model giving you a clear overview of the causal mechanisms the inputs in a prompt entail. For this thesis, we experiment on [GPT-2](https://huggingface.co/openai-community/gpt2). One of the requirements for DAS to find the alignments between a causal model and the neural representations is for the neural network (GPT-2 in our case) to have a high performance on the task it tries to solve.

### Finetuning GPT2 to an arithmetic task

We finetune [GPT2ForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/gpt2#transformers.GPT2ForSequenceClassification) to perform well on summing three numbers between 1 and 10 (including). The prompts are of the form `X+Y+Z=`, where `X`, `Y`, `Z` are inputs randomly generated from 1 to 10. There are 28 possible outputs, and 1000 possible arrangements of the numbers. The task is to classify the sequence generated by GPT2 after `=` into one of the possible 28 classes. To finetune GPT2 on this task, run:

```
python3 train_gpt2.py
```

### Training DAS

We define three possible graphs which can abstract the finetuned GPT2. They are represented in the below graphs, where `P` is the variable summing each pair of two variables. We refer to these graphs as the _arithmetic_ ones.

![](https://raw.githubusercontent.com/maraPislar/LLM_causal_model_learning/main/results/imgs/(X+Y)+Z.png)
![](https://raw.githubusercontent.com/maraPislar/LLM_causal_model_learning/main/results/imgs/(X+Z)+Y.png)
![](https://raw.githubusercontent.com/maraPislar/LLM_causal_model_learning/main/results/imgs/X+(Y+Z).png)

We also define a _simple_ group of causal graphs, where each graph just copies in turns each input variable to an intervenable varibale.

![](https://raw.githubusercontent.com/maraPislar/LLM_causal_model_learning/main/results/imgs/(X)+Y+Z.png)
![](https://raw.githubusercontent.com/maraPislar/LLM_causal_model_learning/main/results/imgs/X+(Y)+Z.png)
![](https://raw.githubusercontent.com/maraPislar/LLM_causal_model_learning/main/results/imgs/X+Y+(Z).png)

There is an intervenable model trained for each of the 12 layers of the LLM, targetting the subspace divided by each of the values in `[64, 128, 256, 768, 4608]`, referred as the low rank dimension. All tokens are targetted. 

To train the each intervenable model using the _arithmetic_ causal models, run:

```
python3 src/run_das.py --model_path /home/mpislar/LLM_causal_model_learning/models/trained_gpt2forseq --causal_model_type arithmetic --n_training 256000 --n_testing 256 --batch_size 1280 --epochs 4
```

To train each intervenable model using the _simple_ causal models, run:

```
python3 src/run_das.py --model_path /home/mpislar/LLM_causal_model_learning/models/trained_gpt2forseq --causal_model_type simple --n_training 256000 --n_testing 256 --batch_size 1280 --epochs 4
```

### Sanity check experiment

After training the intervenable models for each layer and lower rank dimension listed in the previous section, run a sanity check experiment similar to the one in the MLP section. It is sufficient to run the sanity check on the arithmetic causal models.

Example of command to run the sanity check:

```
python3 visualizations.py --model_path /home/mpislar/LLM_causal_model_learning/models/trained_gpt2forseq --causal_model_type arithmetic --results_path disentangling_results/ --experiment sanity_check
```


### Empirical experiment (where does each variable live?)

We want to check where each variable lives when we use intervenable variables which are only copies of the input variables. After training the intervenable models when aligning the simple causal models with the LLM, one can check the IIA per layer and low rank dimension. To reproduce our plots, run this command:

```
python3 visualizations.py --model_path /home/mpislar/LLM_causal_model_learning/models/trained_gpt2forseq --causal_model_type simple --results_path disentangling_results/ --experiment empirical
```

