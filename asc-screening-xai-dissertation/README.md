### Explainable Machine Learning for Autism Screening

#### 

#### Overview



This repository contains the code and experimental materials for an MSc dissertation project investigating the use of explainable machine learning (XAI) methods to support early screening for Autism Spectrum Condition (ASC). The project focuses on methodological rigour, evaluation robustness and interpretability, rather than maximising raw predictive accuracy.

The work uses publicly available adult and child autism screening datasets derived from the Autism Spectrum Quotient (AQ-10) questionnaire and evaluates multiple machine learning pipelines under a leak-safe experimental framework. Explainability is treated as a core component of the study and is implemented using SHAP for the final selected model.

#### 

#### Project Aims



The main aims of this project are to:

Evaluate whether behavioural questionnaire data can support early autism screening using machine learning

Compare different machine learning approaches in terms of performance, robustness and suitability for screening tasks

Integrate explainable machine learning to support transparent and interpretable predictions

Avoid common methodological pitfalls such as data leakage and inflated performance

This project positions machine learning as a decision-support tool rather than a diagnostic system and does not aim to replace clinical judgement.

#### 

#### Datasets



Public datasets from the UCI Machine Learning Repository are used:

* Autism Screening Adult Dataset  
  https://archive.ics.uci.edu/dataset/426/autism+screening+adult
* Autism Screening Child Dataset  
  https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children

#### 

#### Notes



* Models are intended for **screening support only**, not diagnosis.
* Explainability is implemented using SHAP for the final selected model.
* All data used is publicly available and anonymised.
