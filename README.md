# She is an Expert in this Research Field: The Signal of Recent Publications' Relevance
This repository contains the code associated with our research paper titled "She is an Expert in this Research Field: The Signal of Recent Publications' Relevance," which has been published in the Northeast Journal of Complex Systems (NEJCS). The paper is available at the following link: https://orb.binghamton.edu/nejcs/vol5/iss1/7.


## Table of Contents

- [Create_NIH_dataset.ipynb](https://github.com/gilzeevi25/Alpha_Relevance/blob/main/Create_NIH_dataset.ipynb) <br>
  This Jupyter Notebook provides a step-by-step guide on how to recreate the NIH dataset used in the paper. The dataset is an essential component of the research, and this notebook ensures that anyone can reproduce it from scratch. <br>
- [Alpha_Relevance_notebook.ipynb](https://github.com/gilzeevi25/Alpha_Relevance/blob/main/Alpha_Relevance_notebook.ipynb) <br>
In this Jupyter Notebook, you will find the code used to create similarity CSVs. The alpha relevance calculations are crucial for the analysis conducted in the paper. This notebook helps users understand the process and allows them to generate their similarity CSVs. <br>
- [plot_similarities.ipynb](https://github.com/gilzeevi25/Alpha_Relevance/blob/main/plot_similarities.ipynb) <br>
The plot_similarities.ipynb notebook contains code to create the plots presented in the paper. These visualizations are essential for understanding the results and conclusions drawn from the analysis. This notebook provides a clear representation of the data and enables users to reproduce the plots.

## Prerequisites

Before running the notebooks, please make sure you have the following packages installed:
```
!pip install emoji --upgrade
!pip install wordcloud
!pip install spacy
!python -m spacy download en_core_web_lg
!pip install gensim
!pip install sentence_transformers
```
These packages are necessary to ensure the code runs smoothly and reproduces the results presented in the paper.

## Data and Usage

The final processed grant calls and winning applications can be found [here](https://drive.google.com/drive/folders/18CATV14JaZXQDRJq2oKcBE90-vr_QibN?usp=sharing).<br>
As we dont hold the legal rights on the winning PIs publications dataset, we provide the code how to recreate it quite easily.
Go to: <u>Get publications based on scopus ID of extracted PIs</u> section in Create_NIH_dataset.ipynb , load ```winning_submission.csv``` and execute the remaining cells

## Recommended Citation
If you find this work helpful, please make sure to cite:
```
Zeevi, Gil and Mokryn, Osnat (2023) "She is an Expert in this Research Field: The Signal of Recent Publications' Relevance," Northeast Journal of Complex Systems (NEJCS): Vol. 5 : No. 1 , Article 7.
DOI: 10.22191/nejcs/vol5/iss1/7/
Available at: https://orb.binghamton.edu/nejcs/vol5/iss1/7
```
## Contact
For any inquiries or collaboration opportunities, feel free to reach out to us:
[Gil Zeevi](gzeevi25@gmail.com) <br>
[Dr Osnat Mokryn](ossimo@gmail.com)<br>
