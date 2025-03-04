# CrossProbe

The repository contains the code and data for the paper "CrossProbe: LLM-empowered Cross-Project Bug Detection for Deep Learning Frameworks" submitted to ISSTA 2025.

## Data

- `data/` contains the original documentation from the frameworks. The data is collected from the following sources:
    - [PyTorch](https://github.com/pytorch/pytorch/tree/main/docs)
    - [TensorFlow](https://github.com/tensorflow/docs)
- The raw documentation files can avoid HTML tags and other irrelevant information. The data is preprocessed and stored in the `data/processed` directory.

## Environment

- The environment is created using `conda`. To create the environment, run the following command:
    ```bash
    conda create --name <env> python=3.10
    ```
    where `<env>` is the environment name.
    ```

- `environ` contains the dependencies for each experiment environment in the paper. To install the dependencies, run the following command:
    ```bash
    pip install -r environ/<env>.txt
    ```
    where `<env>` is the environment name.

## Test Cases

- `pytorch/` and `tensorflow/` contain the test cases for the respective frameworks, which are collected from the issues and pull requests.

- The file names are in the format `<source_framework>-<issue_id>-<target_framework>.py`.

## TODO

- [ ] Add the Jupyter Notebook for the experiments.
- [ ] Organize the raw data for OpenAI outputs.
