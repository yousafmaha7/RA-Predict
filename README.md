# RA-Predict: Machine Learning-Based Rheumatoid Arthritis Prediction Tool

## Overview

**RA-Predict** is a machine learning-based web application developed for the prediction of rheumatoid arthritis (RA) using clinical and laboratory parameters.

The tool provides a user-friendly interface that allows researchers and healthcare professionals to input patient-related features and obtain rapid RA prediction results using a trained machine learning model.

## Features

* Machine learning-based RA prediction
* User-friendly Streamlit web interface
* Rapid prediction from clinical parameters
* Example testing dataset included
* Pre-trained model integrated for immediate use

## Project Structure

```
RA-Predict/
│
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── assets/
│   └── logo.png
│
├── examples/
│   └── RA_Testing_Data.csv
│
├── model/
│   └── RA_Predict.pkl
│
├── reports/
│
└── build/
```

## Installation

### Clone the repository

```bash
git clone https://github.com/yousafmaha7/RA-Predict.git
cd RA-Predict
```

### Create a virtual environment

#### Ubuntu/macOS

```bash
python3 -m venv ra_env
source ra_env/bin/activate
```

#### Windows

```bash
python -m venv ra_env
ra_env\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to:

```
http://localhost:8501
```

## Usage

1. Enter the required clinical and laboratory parameters.
2. Submit the input data through the application interface.
3. The model will generate an RA prediction result.

An example dataset is provided in:

```
examples/RA_Testing_Data.csv
```

## Model Information

The prediction engine is powered by a trained machine learning model stored at:

```
model/RA_Predict.pkl
```

The model was developed using clinical and laboratory features relevant to rheumatoid arthritis prediction.

## Requirements

The required Python packages are listed in:

```
requirements.txt
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Citation

If you use RA-Predict in your research, please cite 
```
AI Mediated Early Detection of Rheumatoid Arthritis Using Clinical, Biochemical and Demographic Features in Pakistan  

Maha Yousaf1,†, Dur E Nishwa1,†, Saadia Iftikhar2, Shaheen Shahzad2, Zeeshan Abbas3,*, Seung Won Lee1,4,5,6,7* 
```
## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.

