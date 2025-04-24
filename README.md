# FaaSRCA
## Description
_**FaaSRCA**_ is a full lifecycle root cause analysis method for serverless applications. It leverages the _Global Call Graph_ to integrate multi-modal observability data generated from both the platform and application sides. By employing different modalities, we capture the features of individual nodes. _**FaaSRCA**_ trains an auto-encoder based on Graph Attention Network (GAT) to compute reconstruction scores for nodes in the global call graph. We measure the deviation between the scores of each node and its corresponding normal pattern. Finally, _**FaaSRCA**_ ranks the deviations to determine the root causes at the granularity of lifecycle stages of serverless functions.

## Quick Start
### Requirements
- python >= 3.8
- numpy >= 1.24.3
- torch >= 2.0.0
- torch_geometric >= 2.3.0
- ntlk >= 3.8.1 
- transformers >= 4.34.0

### Running
```
1. python /rca/train.py
2. python /rca/normal_score.py
3. python /rca/normal_distribution.py
4. python /rca/root_localization.py
```
### Project Structure
```
├── LICENSE
├── README.md
├── rca 
│   ├── create_dataset.py: create graph dataset
│   ├── train.py: train graph auto-encoder
│   ├── normal_score.py: calculate the scores for nodes under normal pattern
│   ├── normal_distribution.py: calculate the distribution of the scores under normal pattern
│   └── root_localization.py: root cause analysis
├── embedding
│   ├── embedding_model.py: create and init embedding model for metrics/traces
│   ├── sentence_embedding.py: embed logs
│   └── all_embedding.py: fuse multi-modal embeddings
├── rca 
│   ├── classify_graph.py: obtain graph category
│   └── template_preprocess.py: preprocess log template
```
