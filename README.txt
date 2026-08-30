This dataset comprises 1,689 focal-paper observations from three scientific fields: 843 observations from artificial intelligence, 441 observations from physics, and 405 observations from biomedicine. The biomedicine data consist of 287 observations in the Lasker subset and 118 observations in the Nobel subset. The dataset is designed to support the early identification and prediction of scientific breakthroughs using a cusp catastrophe model based on knowledge heterogeneity and knowledge relevance.

Archive structure:
-all data/AI/AI_dataset.xlsx: paper-level metadata for 843 artificial-intelligence papers
-all data/AI/AI_discriminant.xlsx: paper-level variables and cusp catastrophe discriminant values for the artificial-intelligence data
-all data/AI/AI_abstract/: one focal-paper metadata/abstract file and one reference metadata/abstract file for each paper
-all data/AI/AI_citation/: citation sentences and citation-function categories for each paper
-all data/physics/physics_dataset.csv: paper-level metadata for 441 physics papers
-all data/physics/physics_discriminant.csv: paper-level variables and cusp catastrophe discriminant values for the physics data
-all data/physics/physics_abstract/: one focal-paper metadata/abstract file and one reference metadata/abstract file for each paper
-all data/physics/physics_citation/: citation sentences and citation-function categories for each paper
-all data/biomedicine/biomedicine_dataset.csv: combined paper-level metadata for 405 biomedical papers
-all data/biomedicine/biomedicine_discriminant.csv: combined paper-level variables and cusp catastrophe discriminant values for the biomedical data
-all data/biomedicine/lasker/: metadata, abstracts, references, citation sentences, and discriminant values for the 287-paper Lasker subset
-all data/biomedicine/nobel/: metadata, abstracts, references, citation sentences, and discriminant values for the 118-paper Nobel subset

Variables in the paper-level metadata files:
-id: unique identifier in the combined biomedicine data; the prefix indicates the source subset (lasker or nobel)
-idx: paper index within the corresponding field or source subset; this value is not globally unique across all three fields
-journal or Journal: journal or conference of publication
-year, Pubyear, or Pub year: year of publication
-Year: award year, where available
-Author: award recipient, where available
-topic: award topic or citation, where available
-DOI: digital object identifier, where available
-award: award status (1 indicates an award-winning paper; 0 indicates a non-award-winning comparison paper)
-title: title of the paper

Variables in the discriminant files:
-id: unique identifier in the combined biomedicine data
-idx: paper index within the corresponding field or source subset
-journal: journal or conference of publication (included in the artificial-intelligence file)
-year: year of publication (included in the artificial-intelligence file)
-award: award status (1 indicates an award-winning paper; 0 indicates a non-award-winning comparison paper)
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
-m_std: standardized value of m used in the cusp catastrophe model
-n_std: standardized value of n used in the cusp catastrophe model
-discriminant: cusp catastrophe discriminant function value

Variables in the per-paper abstract and reference files:
-paper_id or idx: identifier of the focal paper or reference
-title: paper title
-DOI: digital object identifier, where available
-abstract: paper abstract, where available
-tldr: short summary returned by the source service, where available
-referenceCount: number of references
-citationCount: number of citations
-influentialCitationCount: number of influential citations, where available
-fieldsOfStudy: disciplinary classification, where available
-time: publication year or source time field, where available
-focus_paper_id: identifier of the focal paper associated with a reference
-focus_paper_title: title of the focal paper associated with a reference

Variables in the per-paper citation files:
-title: title of the focal paper
-sentence: sentence in which the focal paper is cited
-category: citation-function depth category. The stored labels are 深度引用 (deep citation), 中度引用 (moderate citation), and 浅度引用 (shallow citation)

File naming and missing values:
-In the abstract directories, <idx>_paper.csv contains focal-paper information and <idx>_ref.csv contains information about its references.
-In the artificial-intelligence citation directory, files are named paper_<idx>.csv. In the physics and biomedicine citation directories, files are named <idx>_paper.csv.
-Missing source-derived values may be represented by empty cells or, in some tldr fields, by the text "undefined".

File encoding:
-Most CSV files are UTF-8; some include a UTF-8 byte-order mark and can be read with the utf-8-sig encoding.
-physics_dataset.csv is encoded in GBK/CP936 and should be read with the gbk encoding to preserve mathematical symbols.