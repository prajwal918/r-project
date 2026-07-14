# Use Rocker's tidyverse image as a solid base
FROM rocker/tidyverse:4.4.1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install required R packages for modeling and ensemble
RUN R -e "install.packages(c('caret', 'randomForest', 'glmnet', 'class', 'rpart', 'rpart.plot', 'e1071', 'gbm', 'xgboost', 'nnet', 'pROC', 'keras', 'tensorflow', 'keras3', 'gridExtra'), repos='http://cran.rstudio.com/')"

# Set the working directory
WORKDIR /app

# Copy the source code and other artifacts
COPY . /app

# Default command to run the meta model ensemble
CMD ["Rscript", "src/meta_model_ensemble.R"]