# NLP Sentiment Analysis Project

A high-throughput NLP API using Django REST to perform sentiment analysis on text input.

This project is an asynchronous Django application for sentiment analysis using a pre-trained RoBERTa model. The application analyzes the sentiment of given text and returns the sentiment label along with a confidence score.

## Table of Contents

- [NLP Sentiment Analysis Project](#nlp-sentiment-analysis-project)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Local Setup](#local-setup)
    - [Docker Setup](#docker-setup)
  - [Usage](#usage)
    - [Local Development](#local-development)
    - [Docker](#docker)
  - [Testing](#testing)
    - [Local Testing](#local-testing)
    - [Docker Testing](#docker-testing)
  - [Deployment](#deployment)
    - [Docker Production Deployment](#docker-production-deployment)
    - [AWS Deployment](#aws-deployment)
  - [Makefile Commands](#makefile-commands)
  - [Justifications](#justifications)

## Installation

### Local Setup

note: You can use the makefile commands to run the commands below

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/nlp-sentiment-analysis.git
    cd nlp-sentiment-analysis
    ```

3. **Setup a virtual environment:**
    ```bash
    pip install -U pipenv
    pipenv shell
    ```

4. **Install dependencies:**
    ```bash
    pipenv install --dev
    ```

5. **Download ML models:**

    Only run this command if you haven't downloaded the models yet to the nlp_sentiment_analysis/ml_models folder
    ```bash
    python nlp_sentiment_analysis/manage.py download_models
    ```

6. **Create and configure the database:**
    - For Linux:
        ```bash
        sudo -u postgres psql -c 'create database your_database_name;'
        sudo -u postgres psql -c 'grant all privileges on database your_database_name to your_database_user;'
        ```
    - For Mac:
        ```bash
        sudo mkdir -p /etc/paths.d && echo /Applications/Postgres.app/Contents/Versions/latest/bin | sudo tee /etc/paths.d/postgresapp
        sudo /Applications/Postgres.app/Contents/Versions/latest/bin/psql -U postgres -c 'create database your_database_name;'
        sudo /Applications/Postgres.app/Contents/Versions/latest/bin/psql -U postgres -c 'grant all privileges on database your_database_name to your_database_user;'
        ```

    My apologies for not including Windows instructions. I do not have a Windows machine to test the commands on. I will try to include Windows instructions in the future.

### Docker Setup

1. **Build and run the Docker containers:**
    ```bash
    docker-compose up -d --build
    ```

2. **Download ML models:**
    Only run this command if you haven't downloaded the models yet to the nlp_sentiment_analysis/ml_models folder
    ```bash
    docker-compose exec backend python nlp_sentiment_analysis/manage.py download_models
    ```

## Usage

### Local Development

1. **Run migrations and start the development server:**
    ```bash
    python nlp_sentiment_analysis/manage.py migrate
    python nlp_sentiment_analysis/manage.py runserver
    ```

### Docker

1. **Build and run the Docker containers:**
    ```bash
    docker-compose up -d --build
    ```

2. **View logs:**
    ```bash
    docker-compose logs -f
    ```

## Testing

### Local Testing

1. **Run tests:**
    ```bash
    python nlp_sentiment_analysis/manage.py test text_analysis
    ```

### Docker Testing

1. **Run tests inside Docker:**
    ```bash
    docker-compose exec backend python nlp_sentiment_analysis/manage.py test text_analysis
    ```

## Deployment

### Docker Production Deployment

1. **Run migrations and start the production containers:**
    ```bash
    docker-compose -f docker-compose.prod.yml up -d --build
    ```

2. **Download ML models:**
    ```bash
    docker-compose -f docker-compose.prod.yml exec backend python nlp_sentiment_analysis/manage.py download_models
    ```

### AWS Deployment
Incompleete instructions. I will update this section with the correct instructions in the future.

1. **Build and push the Docker image to AWS ECR:**
    ```bash
    docker build -t your_aws_account_uri:latest .
    aws ecr get-login-password --region your_aws_region | docker login --username AWS --password-stdin your_aws_account_id.dkr.ecr.your_aws_region.amazonaws.com
    docker push your_aws_account_id.dkr.ecr.your_aws_region.amazonaws.com/django-app:latest
    ```

## Makefile Commands

- **help:** Show available targets.
- **venv:** Setup virtual environment.
- **install-packages:** Install project dependencies.
- **create-local-database-linux:** Create a local database on Linux.
- **create-local-database-mac:** Create a local database on Mac.
- **drop-local-database-linux:** Drop a local database on Linux.
- **drop-local-database-mac:** Drop a local database on Mac.
- **run-local:** Run migrations and start the local development server.
- **migrate:** Run database migrations.
- **test:** Run tests.
- **up:** Build and start Docker containers.
- **down:** Stop and remove Docker containers.
- **logs:** View Docker container logs.
- **test-docker:** Run tests inside Docker containers.
- **prod-migrate:** Run migrations in production.
- **prod-download-ml-models:** Download ML models in production.
- **prod-up:** Build and start production Docker containers.
- **prod-down:** Stop and remove production Docker containers.
- **copy-env:** Copy environment example file.
- **push-image-aws:** Build and push Docker image to AWS ECR.

For detailed usage of each command, refer to the `Makefile` provided in the project.

## Justifications

- **Django REST Framework**: Django REST Framework is a powerful and flexible toolkit for building Web APIs. It is a popular choice for building APIs in Django projects. It is well-documented and has a large community, which makes it easy to find solutions to common problems.
- **OpenAPI**: OpenAPI is a specification for building APIs. It provides a standard way to describe APIs, which makes it easier to understand and work with APIs. It also provides tools for generating client libraries and documentation, which can be very useful when building and consuming APIs.
- **TensorFlow**: TensorFlow is a popular machine learning library that provides tools for building and training deep learning models. It is widely used in the machine learning community and has a large number of pre-trained models that can be used for various tasks.
- **Docker**: Docker is a popular tool for building, shipping, and running applications in containers. It provides a lightweight and portable way to package applications and their dependencies, which makes it easy to deploy applications in different environments.
- **cardiffnlp/twitter-roberta-base-sentiment**: This is a pre-trained RoBERTa model that has been fine-tuned on a sentiment analysis task. It is a powerful model that can be used to perform sentiment analysis on text input.
- **transformers**: Transformers is a popular library that provides tools for working with pre-trained models like RoBERTa. It provides an easy way to load and use pre-trained models, which makes it easy to build and deploy machine learning models.
- **Redis**: Redis is an in-memory data store that can be used to cache data and speed up applications. It is a popular choice for caching data in web applications and can be used to store the results of sentiment analysis requests.

References:

- [<https://medium.com/@sharma.tanish096/sentiment-analysis-using-pre-trained-models-and-transformer-28e9b9486641>]
- [https://huggingface.co/docs/transformers/v4.40.2/en/model_doc/auto#transformers.TFAutoModelForSequenceClassification]
- [A Gentle Introduction to RoBERTa](https://www.analyticsvidhya.com/blog/2022/10/a-gentle-introduction-to-roberta/)
- [PyTorch vs TensorFlow for Transformer-based NLP Applications](https://towardsdatascience.com/pytorch-vs-tensorflow-for-transformer-based-nlp-applications-b851bdbf229a)
- [An Explanatory Guide to BERT Tokenizer](https://www.analyticsvidhya.com/blog/2021/09/an-explanatory-guide-to-bert-tokenizer/)
