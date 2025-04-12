# CrossProbe

The repository contains the code and data for the paper "CrossProbe: LLM-empowered Cross-Project Bug Detection for Deep Learning Frameworks" submitted to ISSTA 2025.

## Getting started

### Environment

- The environment is created using `conda`. To create the environment, run the following command:
    ```bash
    conda create --name <env> python=3.10
    conda activate <env>
    conda install ipykernel ipywidgets
    ```
    where `<env>` is the environment name.

- The required packages are listed in notebooks. To install them in advance, run the following command:
    ```bash
    conda activate <env>
    pip install transformers torch 'numpy<2' pandas openai
    ```

- To use OpenAI models, the OpenAI API key is required. You can set the API key in the environment variable `OPENAI_API_KEY` or directly in the Jupyter Notebook.

### Step 1: Documentation processing

- The data is preprocessed with `docs.ipynb`.
    - Open the Jupyter Notebook and run the cells to process the documentation.
- The API documentation database is stored in `api_documentation_db.csv`.

### Step 2: Code generation

- The code generation is done with `generation.ipynb`.
    - Open the Jupyter Notebook and run the cells to generate the test cases.
- The generated test cases are stored in `pytorch-test` and `tensorflow-test` folders.

## Detailed description

### Data source

- `data/` contains the original documentation from the frameworks. The data is collected from the following sources:
    - [PyTorch](https://github.com/pytorch/pytorch/releases/tag/v2.4.0)
    - [TensorFlow](https://github.com/tensorflow/tensorflow/releases/tag/v2.17.0)
- The raw documentation files can avoid HTML tags and other irrelevant information.

### Generated test cases

- `pytorch/` and `tensorflow/` contain the test cases for the respective frameworks, which are collected from the issues and pull requests.
    - The file names are in the format `<source_framework>-<issue_id>-<target_framework>.py`.
