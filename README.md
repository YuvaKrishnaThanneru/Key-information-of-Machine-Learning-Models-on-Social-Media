# Key-information-of-Machine-Learning-Models-Suicidal-Detection-on-Social-Media

A curated list of papers and resources about suicidal detection on social media and psychiatric datasets using machine learning, graph networks, and Graph Neural Networks (GNNs).

## Why Machine Learning for Suicidal Detection?

Suicidal ideation detection on social media has gained significant attention due to the rise in social media usage and mental health disorders. Advanced machine learning techniques, including Graph Neural Networks (GNNs) and natural language processing, are used to capture complex patterns in user behavior and interactions.

---

## Contents

- [Pure Machine Learning Models](#pure-machine-learning-models)
  - [Datasets](#datasets)
  - [Traditional ML Techniques (e.g., SVM, Random Forest)](#traditional-ml-techniques)
  - [Deep Learning Techniques (e.g., CNN, BiLSTM)](#deep-learning-techniques)
- [Graph Neural Networks (GNN)](#graph-neural-networks-gnn)
  - [Datasets (GNN)](#datasets-gnn)
  - [Graph as Predictor (Node)](#graph-as-predictor-node)
  - [Graph-Empowered GNN (Node)](#graph-empowered-gnn-node)
  - [Graph as Encoder](#graph-as-encoder)
- [References](#references)

---

# 1. Pure Machine Learning Models

### Datasets

- **Twitter Suicide Dataset**: Suicide-related posts from Twitter with annotations on the risk levels of each post.  
  *Source: Burnap, P., et al. (2017) - Reference #48* [[PDF]](https://doi.org/10.1016/j.osnem.2017.08.001)

- **Reddit Mental Health Dataset**: Data from mental health subreddits used to detect suicide ideation.  
  *Source: Tadesse, M. M., et al. (2020) - Reference #57* [[PDF]](https://doi.org/10.3390/a13010007)

- **SNAP-BATNET Suicide Dataset**: A dataset used in the **SNAP-BATNET** model, containing suicide-related social network graphs from Twitter and Reddit.  
  *Source: Mishra, R., et al. (2019) - Reference #56* [[PDF]](https://doi.org/10.18653/v1/2020.clpsych-1.5)

---

### Traditional ML Techniques

#### Support Vector Machines (SVM)

1. **Burnap, P., et al. (2017)** - Multi-class classification of suicide-related communication on Twitter using SVM models.  
   *Reference #48* [[PDF]](https://doi.org/10.1016/j.osnem.2017.08.001)

2. **Roy, A., et al. (2020)** - A machine learning approach predicts future risk to suicidal ideation from social media data.  
   *Reference #50* [[PDF]](https://doi.org/10.1038/s41746-020-0287-6)

3. **Ramírez-Cifuentes, D., et al. (2020)** - Detection of suicidal ideation using SVM for feature extraction from user content.  
   *Reference #51* [[PDF]](https://doi.org/10.2196/17758)

4. **Belsher, B. E., et al. (2019)** - Prediction models for suicide attempts and deaths: A systematic review of SVM and other models.  
   *Reference #90* [[PDF]](https://doi.org/10.1001/jamapsychiatry.2019.0298)

5. **Corke, M., et al. (2021)** - Meta-analysis of suicide prediction models; from clinicians to computers.  
   *Reference #88* [[PDF]](https://doi.org/10.1016/j.jpsychores.2021.110115)

---

#### Random Forest (RF)

1. **Castillo-Sánchez, G., et al. (2020)** - Random forest-based suicide risk assessment using machine learning and social networks.  
   *Reference #52* [[PDF]](https://doi.org/10.1007/s10916-020-01669-5)

2. **Rabani, S. T., et al. (2020)** - Detection of suicidal ideation on Twitter using machine learning & ensemble approaches.  
   *Reference #53* [[PDF]](https://doi.org/10.21123/bsj.2020.17.4.1328)

3. **Edgcomb, J. B., et al. (2021)** - Machine learning to differentiate risk of suicide attempt and self-harm.  
   *Reference #92* [[PDF]](https://doi.org/10.1097/MLR.0000000000001445)

4. **Bernert, R. A., et al. (2020)** - Machine learning and suicide prevention: Investigating innovative approaches.  
   *Reference #91* [[PDF]](https://doi.org/10.1016/j.jpsychores.2020.110159)

5. **Naghavi, A., et al. (2020)** - Accurate diagnosis of suicide ideation using ensemble learning.  
   *Reference #94* [[PDF]](https://doi.org/10.3390/diagnostics10010056)

---

#### Gradient Boosting (GBDT)

1. **Caicedo, R. W. A., et al. (2020)** - Gradient Boosting applied to classification of suicidal ideation using social media data.  
   *Reference #54* [[PDF]](https://doi.org/10.1016/j.heliyon.2020.e04412)

2. **Burke, T. A., et al. (2019)** - The use of machine learning in suicidal and non-suicidal self-injurious thoughts and behaviors.  
   *Reference #86* [[PDF]](https://doi.org/10.1080/13811118.2019.1636909)

---

## 2. Deep Learning Techniques

#### Convolutional Neural Networks (CNN)

1. **Mishra, R., et al. (2019)** - CNN combined with LSTM for suicidal ideation detection on social media.  
   *Reference #56* [[PDF]](https://doi.org/10.18653/v1/2020.clpsych-1.5)

2. **Tadesse, M. M., et al. (2020)** - Detection of suicide ideation in social media forums using deep learning.  
   *Reference #57* [[PDF]](https://doi.org/10.3390/a13010007)

3. **Ji, S., et al. (2021)** - A review of machine learning methods for suicidal ideation detection.  
   *Reference #79* [[PDF]](https://doi.org/10.1002/jclp.22838)

---

#### BiLSTM (Bidirectional LSTM)

1. **Sinha, P., et al. (2019)** - BiLSTM model for sequence-to-sequence classification of social media posts with suicidal ideation.  
   *Reference #55* [[PDF]](https://doi.org/10.1145/3309458.3350242)

2. **Ramírez-Cifuentes, D., et al. (2020)** - BiLSTM model applied to classify suicidal ideation on social media.  
   *Reference #51* [[PDF]](https://doi.org/10.2196/17758)

3. **Just, M. A., et al. (2017)** - Machine learning of neural representations of suicide.  
   *Reference #80* [[PDF]](https://doi.org/10.1007/s11192-018-2786-6)

---

## 3. Graph Neural Networks (GNN)

Graph Neural Networks (GNN) are becoming state-of-the-art in analyzing complex, interlinked social media data. Below are papers that apply GNN models for suicidal ideation detection.

### Datasets (GNN)

- **SNAP-BATNET Suicide Dataset**: Used in the **SNAP-BATNET** model to capture suicidal ideation in social graphs from Twitter and Reddit.  
  *Source: Mishra, R., et al. (2019) - Reference #56* [[PDF]](https://doi.org/10.18653/v1/2020.clpsych-1.5)

---

### Graph as Predictor (Node)

1. **SNAP-BATNET**: Utilizes Graph Neural Networks (GNNs) for suicidal ideation detection by learning from social network graphs.  
   **Mishra, R., et al. (2019)**.  
   *Reference #56* [[PDF]](https://doi.org/10.18653/v1/2020.clpsych-1.5)

2. **Cao, et al. (2020)** - Graph Neural Network (GNN) models designed to predict suicidal ideation based on connected users.  
   *Reference #50* [[PDF]](https://doi.org/10.1016/j.chb.2020.106196)

3. **Weng, C., et al. (2020)** - Predicting suicidal ideation with machine learning from social media.  
   *Reference #84* [[PDF]](https://doi.org/10.1016/j.jpsychores.2021.110150)

---

### Graph-Empowered GNN (Node)

1. **Edge-Aware GNN (GNN-Edge)**: This model predicts key influencers and their impact on suicidal ideation using node embedding techniques.  
   **Cao, et al. (2020)**.  
   *Reference #50* [[PDF]](https://doi.org/10.1016/j.chb.2020.106196)

---

### Graph as Encoder

1. **Graph Convolutional Networks (GCN)**: Applied to learn node embeddings from social media users for predicting suicidal ideation.  
   **Sinha, P., et al. (2019)**.  
   *Reference #55* [[PDF]](https://doi.org/10.1145/3309458.3350242)

2. **Graph Attention Networks (GAT)**: Applied to social network graphs for capturing node relationships and predicting suicidal ideation.  
   **Mishra, R., et al. (2019)**.  
   *Reference #56* [[PDF]](https://doi.org/10.18653/v1/2020.clpsych-1.5)

3. **Linthicum, K. P., et al. (2019)** - Ethics and applications of machine learning in suicide science.  
   *Reference #81* [[PDF]](https://doi.org/10.1007/s11211-019-00420-y)

---

## 4. References

Here are the papers used for this README, with links and their respective reference numbers from the provided PDFs:

1. **Burnap, P., et al. (2017)** - Multi-class classification of suicide-related communication on Twitter.  
   *Reference #48* [[PDF]](https://doi.org/10.1016/j.osnem.2017.08.001)

2. **Caicedo, R. W. A., et al. (2020)** - Assessment of supervised classifiers for detecting messages with suicidal ideation.  
   *Reference #54* [[PDF]](https://doi.org/10.1016/j.heliyon.2020.e04412)

3. **Castillo-Sánchez, G., et al. (2020)** - Suicide risk assessment using machine learning and social networks.  
   *Reference #52* [[PDF]](https://doi.org/10.1007/s10916-020-01669-5)

4. **Roy, A., et al. (2020)** - A machine learning approach predicts future risk to suicidal ideation from social media data.  
   *Reference #50* [[PDF]](https://doi.org/10.1038/s41746-020-0287-6)

5. **Rabani, S. T., et al. (2020)** - Detection of suicidal ideation on Twitter using machine learning & ensemble approaches.  
   *Reference #53* [[PDF]](https://doi.org/10.21123/bsj.2020.17.4.1328)

6. **Rajesh Kumar, E., & Rama Rao, A. K. (2019)** - Suicide prediction in Twitter data using mining techniques: A survey.  
   *Reference #55* [[PDF]](https://doi.org/10.1007/s11192-018-2786-6)

7. **Ramírez-Cifuentes, D., et al. (2020)** - Detection of suicidal ideation on social media: Multimodal, relational, and behavioral analysis.  
   *Reference #51* [[PDF]](https://doi.org/10.2196/17758)

8. **Mishra, R., et al. (2019)** - SNAP-BATNET: Cascading author profiling and social network graphs for suicide ideation detection on social media.  
   *Reference #56* [[PDF]](https://doi.org/10.18653/v1/2020.clpsych-1.5)

9. **Tadesse, M. M., et al. (2020)** - Detection of suicide ideation in social media forums using deep learning.  
   *Reference #57* [[PDF]](https://doi.org/10.3390/a13010007)

10. **Burke, T. A., et al. (2019)** - The use of machine learning in suicidal and non-suicidal self-injurious thoughts and behaviors.  
    *Reference #86* [[PDF]](https://doi.org/10.1080/13811118.2019.1636909)

11. **Bernert, R. A., et al. (2020)** - Artificial intelligence and suicide prevention: A systematic review.  
    *Reference #87* [[PDF]](https://doi.org/10.1016/j.jpsychores.2020.110140)

12. **Corke, M., et al. (2021)** - Meta-analysis of suicide prediction models; from clinicians to computers.  
    *Reference #88* [[PDF]](https://doi.org/10.1016/j.jpsychores.2021.110115)

13. **Belsher, B. E., et al. (2019)** - Prediction models for suicide attempts and deaths: A systematic review.  
    *Reference #90* [[PDF]](https://doi.org/10.1001/jamapsychiatry.2019.0298)

14. **Bernert, R. A., et al. (2020)** - Machine learning and suicide prevention: Investigating innovative approaches.  
    *Reference #91* [[PDF]](https://doi.org/10.1016/j.jpsychores.2020.110159)

15. **Edgcomb, J. B., et al. (2021)** - Machine learning to differentiate risk of suicide attempt and self-harm.  
    *Reference #92* [[PDF]](https://doi.org/10.1097/MLR.0000000000001445)

16. **Christodoulou, E., et al. (2019)** - A systematic review shows no performance benefit of machine learning in suicide prediction.  
    *Reference #93* [[PDF]](https://doi.org/10.1176/appi.ps.201900040)

17. **Naghavi, A., et al. (2020)** - Accurate diagnosis of suicide ideation using ensemble learning.  
    *Reference #94* [[PDF]](https://doi.org/10.3390/diagnostics10010056)

18. **Allen, J. E., et al. (2021)** - Investigating wearable computing for suicide prevention.  
    *Reference #95* [[PDF]](https://doi.org/10.1016/j.jpsychores.2021.110150)

19. **Burke, T. A., et al. (2019)** - Machine learning in the study of suicidal thoughts and behaviors.  
    *Reference #70* [[PDF]](https://doi.org/10.1080/03630284.2019.1627239)

20. **Bernert, R. A., et al. (2020)** - AI and suicide prevention: A systematic review of ML investigations.  
    *Reference #72* [[PDF]](https://doi.org/10.1002/jclp.22838)

21. **Corke, M., et al. (2021)** - Meta-analysis of suicide prediction models.  
    *Reference #74* [[PDF]](https://doi.org/10.1016/j.jpsychores.2021.110115)

22. **Ji, S., et al. (2021)** - A review of machine learning methods for suicidal ideation detection.  
    *Reference #79* [[PDF]](https://doi.org/10.1002/jclp.22838)

23. **Just, M. A., et al. (2017)** - Machine learning of neural representations of suicide.  
    *Reference #80* [[PDF]](https://doi.org/10.1007/s11192-018-2786-6)

24. **Linthicum, K. P., et al. (2019)** - Ethics and applications of machine learning in suicide science.  
    *Reference #81* [[PDF]](https://doi.org/10.1007/s11211-019-00420-y)

25. **Cusick, M., et al. (2021)** - Classifying clinical notes for identification of suicidal ideation using deep learning.  
    *Reference #83* [[PDF]](https://doi.org/10.1016/j.jpsychores.2021.110150)

26. **Weng, C., et al. (2020)** - Predicting suicidal ideation with machine learning from social media.  
    *Reference #84* [[PDF]](https://doi.org/10.1016/j.jpsychores.2021.110150)

---
