# Weka Model Runner



This project provides a Java-based CLI for training and testing various machine learning models (J48, RandomForest, MLP, SGD, RBF, XGBoost) with their default parameters using Weka. Users can run it either directly with the JAR file or via Docker.

By default, the provided dataset is split into **80% training** and **20% testing**, and evaluation metrics are computed on the test set.


## Running with JAR



1\. Build your project and generate the JAR using Maven:



```bash

mvn clean package

```



2\. Navigate to the `target` directory:



```bash

cd target

```



3\. Run the JAR with the following syntax:



```bash

java -jar weka-model-runner.jar <model> <datasetKey|path> <targetIndex> <taskType>

```



* `<model>`: One of `J48`, `RandomForest`, `MLP`, `SGD`, `RBF`, `XGBoost`

* `<datasetKey|path>`: Either a predefined dataset key (`tennis`, `iris`, `wine`) or a full path to your CSV file

* `<targetIndex>`: Index of the target column (0-based, `-1` for last column)

* `<taskType>`: `cls` for classification or `reg` for regression



Example:



```bash

java -jar weka-model-runner.jar J48 tennis -1 cls

```



## Running with Docker



1\. Build the Docker image (make sure Docker is installed and running):



```bash

docker build -t weka-model-runner .

```



2\. Run the container with the same arguments as the JAR:



```bash

docker run --rm weka-model-runner <model> <datasetKey|path> <targetIndex> <taskType>

```



Example:



```bash

docker run --rm weka-model-runner J48 tennis -1 cls

```



> The `--rm` flag ensures the container is removed after it finishes running.



## Predefined Datasets



* `tennis`

* `iris`

* `winequality-red`



You can also provide the path to a custom CSV dataset.

## Download Pre-Built JAR

You can also download the latest release JAR from [GitHub Releases](https://github.com/kteleg/weka-model-runner/releases) and run it without building the project.

## Notes



* Ensure your CSV files are correctly formatted with headers.

* Target column index starts at 0. Use `-1` for the last column.

* Classification models require nominal targets, while regression requires numeric targets.

* Docker runs a self-contained environment, so you do not need Java installed on the host.

* JAR execution requires Java installed on your system.



