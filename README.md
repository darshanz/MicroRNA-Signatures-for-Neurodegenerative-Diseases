# MicroRNA Signatures for Neurodegenerative Diseases


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Active%20Development-orange)
 
This is an unofficial implementation of the study by Li et al. (2022) 
titled "[Identifying Key MicroRNA Signatures for Neurodegenerative Diseases 
With Machine Learning
 Methods](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2022.880997/full)" 
 published in *Frontiers in Genetics*.
 
 Disclaimer: This implementation is not affiliated with or endorsed by the original authors 
 and was implemented purely for study and educational purposes.

 ---

This repository contains code and analysis for identifying microRNA signatures 
associated with various neurodegenerative diseases using publicly available
 gene expression datasets. 



 ### Dataset

The dataset used in this study is sourced from the Gene Expression Omnibus (GEO) download [GEO Accession viewer](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120584).

The dataset contains 1601 samples for various neurodegenerative diseases as shown in the table below (01B_Data_Exploration.ipynb):


| Disease Case | Sample Size |
|--------------|-------------|
| Alzheimer’s disease (AD) | 1,021 |
| Vascular dementia (VaD) | 91 |
| Dementia with lewy bodies (DLB) | 169 |
| Mild cognitive impairment (MCI) | 32 |
| Normal control (NC) | 288 |


### Preprocessing

Preprocessing

..



### File information:

 Repository consists of following notebooks:
 1. **01A_Understanding_Raw_Data.ipynb** : Understanding raw data
 2. **01B_Data_Exploration.ipynb** : series matrix data exploration
 3. **02A_BorutaFeatureRanking.ipynb**: feature filtering using`BorutaPy`
 4. **02B_FeatureRanking.ipynb**: 



### Install Requirements

Requirements for this project are available in `requirements.txt`


 ## References
 [1] Li, Z., Guo, W., Ding, S., Chen, L., Feng, K., Huang, T. and Cai, Y.D., 2022. Identifying key MicroRNA signatures for neurodegenerative diseases with machine learning methods. Frontiers in Genetics, 13, p.880997.

 [2] Kursa M., Rudnicki W., "Feature Selection with the Boruta Package" Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010