# Data Understanding


The ultimate goal is to understand this paper: [Identifying Key MicroRNA Signatures for Neurodegenerative Diseases With Machine Learning Methods](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2022.880997/full) by Li et al. [1],  and replicate the methodlogy in that paper using python. 

However, they appear to be starting their analysis from preprocessed data on [GEO]([GEO Accession viewer](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120584)). 


As our goal is to understand the raw data too. we will do separate analysis on raw data in `Raw_data_exploration.ipynb` to understand the raw data.

Then we continue using same methodology as the paper [1], starting from the series matrix data in `Series_Data_exploration.ipynb`.


# Understanding_Raw_Data.ipynb

In this notebook we processed raw Agilent miRNA microarray data from GEO accession GSE120584 to create an expression matrix for machine learning analysis of neurodegenerative diseases.

The raw data was donwloded from [GEO Accession viewer](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120584) conains compresed txt files. 

Data Information:
 
- **GEO Accession**: GSE120584
- **Technology**: Agilent miRNA Microarray
- **Samples**: 1,601 human serum samples
- **Disease Classes**: AD, VaD, DLB, MCI, and Normal Control

We followed following steps:

1. We extraced them in the separated txt files.
Each files contains:

- **1601 separate .txt files** (one per sample)
- **File format**: Tab-separated with 7-line header
- **Key columns**: 
  - `G_Name`: miRNA names (e.g., "hsa-miR-28-3p")
  - `635nm`: Expression values (Channel 1 measurements)
  - `Flag_635`: Quality flags (all "OK" - high quality data)

2. Upon visual inspection it appeared that the text files have similar format with tab separated values. 
3. The dataframes in each file was found to contain same shape (3200, 16) and identical column names: `['Cell', 'Block', 'Column', 'Row', 'G_Name', 'G_ID', '635nm', ' nm', '635nm.1', ' nm.1', 'Flag_635', 'Flag_ ', 'Flag_635.1', 'Flag_ .1', 'Flag_635.2', 'Flag_ .2']`
4.  Confirmed all quality flags were "OK" (no filtering needed)
5. Matrix Construction
- Combined all 1,601 samples into unified expression matrix
- **Rows**: 2,613 unique miRNAs
- **Columns**: 1,601 samples with informative IDs (e.g., "GSM3404971_MCI_0021")
- **Values**: miRNA expression levels from 635nm channel

## Final Output
 
Expression Matrix Shape: (2613, 1601)
- miRNAs: 2,613 features
- Samples: 1,601 observations
- Missing Values: 0 (complete data)
 
We created a expression matrix with 2,613 mRNAs . The paper [1] mentioned. ` "a total of 2547 miRNAs were identified in the expression profiles"` This difference could be due to different filtering. However, our goal here was to understand the raw data folder.  We deal with the preprocessed matrix following the paper [1] to continue the study.


# Data_exploration.ipynb

