This dataset comprises 1,689 observations from three fields: 843 from artificial intelligence, 441 from physics, and 405 from biomedicine. It is designed to support the early identification and prediction of scientific breakthroughs using a cusp catastrophe model.

Files:
-*_dataset files contain paper-level metadata.
-*_discriminant files contain the variables used in the cusp catastrophe analysis.

Variables:
-id: unique identifier in the combined biomedicine dataset
-idx: paper identifier within each field or subset
-journal: journal or conference of publication
-year: year of publication
-award: award status (1 indicates an award-winning paper; 0 indicates a non-award-winning paper)
-title: title of the paper
-num_all: total number of citation sentences
-num_d: number of deep citation sentences
-num_m: number of moderate citation sentences
-num_l: number of shallow citation sentences
-P_d: proportion of deep citation sentences
-P_m: proportion of moderate citation sentences
-P_l: proportion of shallow citation sentences
-num_refs: total number of references
-mean_similarity: mean semantic similarity between the focal paper and its references
-max_similarity: maximum semantic similarity between the focal paper and its references
-min_similarity: minimum semantic similarity between the focal paper and its references
-std_similarity: standard deviation of semantic similarity between the focal paper and its references
-m: measure of knowledge heterogeneity
-n: measure of knowledge relevance
-m_std: standardized value of m
-n_std: standardized value of n
-discriminant: cusp catastrophe discriminant function value
