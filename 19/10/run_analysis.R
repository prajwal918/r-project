# Quick Script to Install Required Packages and Run Analysis
# Run this first if you haven't already

# Check and install required packages
required_packages <- c("tidyverse", "ggplot2", "caret", "randomForest", "pROC")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

cat("✅ All packages installed!\n")
cat("Now run your main script: er.R\n")
