Distributional DCA Features for Protein-protein Protein–protein interaction site prediction

#Abstract

Protein-protein interaction site investigation is one of the most important fields of biological research in an attempt to identify and catalog physical interactions between pairs or groups of proteins for gaining insights into the fundimental biochemiscal machanisms.
Considering the great cost on time and finance of the conventional biological experiments, many studies emerge in these years focusing on PPI site prediction via the computational approachs.
This paper proposes a novel deeplearning pipeline for identifying the site of PPI. This approach extends previous sucesful methods of deep learning including DeepPPISP and ACCN. We further integrate the DCA features into the framework.
Comparative experiments are conducted on three different public datasets. On various prediciton metrics, our pipeline exhibits significantly better prediction ability compared to other popular PPI prediction methods including PSIVER, SPIDER, ISIS, DEEPPPISP and ACCN. Our predicting accuracy is 10 percentage higher than ACCN which is the second place in performance.

#Introduction

- 生物学价值
- literature review on related computational approaches.
- 我们用到的特征：AA，PSSM，DSSP
- Inspired by alphafold, we introduced DCA.
- Usually, techinique of sequence completion or rough truncation would be applied for structruing the proteins with different length. The former method is very computational time consumed while the latter one will sacrifices a certain amount of information. We redefine the DCA features from adistributional perspective, so that the sequence alignment can be efficiently completed while almost all the information can be retained. 
- to evaluate the performance of our framework, we performed six computational methods (...) on three public dataset (...). In result, our method achieves the best performance for the PPI site prediction in all the metrics.

#Materials and Methods

- Datasets
- Data Process and Feature Selection
	- Sequence Alignment
	- AA
	- PSSM
	- DSSP
	- DDCA
- Network Architecture (Including )
	- subnet1
	- subnet2
	- subnet3
- Evaluation Metrics

#Result and Discussion

- Parameter optimization
- Performance comparisons with other methods
to evaluate the prediction performance of our framework, we performed the methods of PSIVER, SPIDER, ISIS, DEEPPPISP and ACCN on three datasets.
- Effect of distributional transformation of DCA feature on results
we compared the DCA feature processing with or without the transformation in the form of a distribution.
- Ablation
- Sliding windows

#Conclusion

#Supplementary







