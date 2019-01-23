# Pattern_spotting
Proposed solution to the problem of returning the relevant pages (those in which there is, at least, one occurrence) for a given query: http://spotting.univ-rouen.fr/

The proposed solution is to generate regions of interest for each image that correspond to the area of an object candidate, using the Selective Search algorithm by Jasper R. R. Uijlings, Koen E. A. van de Sande, Theo Gevers, Arnold W. M. Smeulders. The candidates, which are massive in number, are filtered with a SVM with RBF Kernel, in order to clean text and backgrounds. For the final candidates fc7 features of the CaffeNet are generated and these are ultimately used to compute simple Euclidean distance to look for similar elements.
