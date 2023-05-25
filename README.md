# Counterfactuals of Counterfactuals
## _A back-translation-inspired approach to analyse counterfactual editors_

![](https://img.shields.io/pypi/pyversions/stanfordnlp.svg?colorB=47be1f)

## Installation

1. Clone the repository.
2. Download and install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
3. Create a Conda environment.
     ```
    conda create -n counter_inco python=3.7
     ```
4. Activate the environment.
     ```
    conda activate counter_inco
     ```
5. Download the requirements.
    ```
    pip3 install -r requirements.txt
    ```
6. For Perplexity:
    ```
    clone https://github.com/jvasilakes/disentanglement-vae.git
    pip install -r /disentanglement-vae/requirements.txt
    ```

## Data Format 
- The structure of the folder where the data should be stored is as follows:
 ```
Edits
└───Dataset1
│   │   Dataset1_editor1.pickle
│   │   Dataset1_editor2.pickle
└───Dataset2
    │   Dataset2_editor1.pickle
    │   Dataset2_editor2.pickle
```
- Edit for each text are saved as lists of lists with the ```[[text1, prob1], [edit_of_test1, new_prob1], ...,]``` e.g.
```
[
    [["I love this movie", [0.99, 0.01]], ["I hate this move", [0.01, 0.99]], ...],
    [["Zero stars!", [0.1, 0.9]], ["Ten stars!", [0.8, 0.2], ...], 
    ...
]
```
In the above example there are 2 different source texts ("I love this movie", "Zero stars!"). The source text and their edits are represented one after each other (for every feed-back step) in each list along with the predictions of the classifier. 

For the first example, the output of the classifier for the source text is ``` [0.99, 0.01]```. The output of the editor of this text is ```"I hate this move"``` and the predicition of the classifier is ```[0.01, 0.99]```. Then the next item of this list should contain the output of the editor for the text: "I hate this move" along with its prediction. 
## Custom Editor - Importing your Data

In order to highlight how a new-custom editor, we provide a script of how we applied an iterative feedback approach to Polyjuice. Following the same practice, we can develop the iterative data for a new editor, as we did with MiCE and Textfooler.
This script also needs to install [Polyjuice](https://github.com/tongshuangwu/polyjuice) and [MiCE](https://github.com/allenai/mice) dependencies.*

This script produces a ```.pickle``` file as the one that is described in the above section. 
## Metrics:
- Minimality
- Inconsistency of Minimality
- Perplexity
-- Base Perplexity: using a proxy model such as GPT-2 
-- Fine Perplexity: unsing a fine tuned version of a proxy model such as fine-tuned GPT-2
-- Probabilities (Notebook)

## Calculate Metrics
- Minimality 
```
>>> python minimality.py  Edits Metrics/Minimality
```
- Inconsistency of Minimality 
```
>>> python inconsistency.py  Edits Metrics/Inconsistency
```
- Base Perplexity
```
>>> # for IMDb
>>> python perplexity.py imdb gpt2 "Edits" "Metrics/Base Perplexity"
>>> # for Newsgroups
>>> python perplexity.py newsgroups gpt2 "Edits" "Metrics/Base Perplexity"
```

- Fine Perplexity
```
>>> # For IMDb
>>> python perplexity.py imdb lvwerra/gpt2-imdb "Edits" "Metrics/Fine Perplexity"
>>> # For Newsgroups
>>> python perplexity.py newsgroups QianWeiTech/GPT2-News "Edits" "Metrics/Fine Perplexity"
```


## Demo & Figures 
We also provide a notebook where we can visualize the edits of the models at different steps. Also, in this notebook, **there are all the results and the figures that are presented in our paper**. The data of these results/figures are produced through the above scripts. 

The results of the custom data/editor could be loaded just by simply running these scripts on custom edits. 


## License
MIT
