# Hate Speech Counterspeech Generator

Use a GPT-2 model to generate counterspeech for hate speech inputs. 

## Features
- Processes hate speech from a CSV file.
- Generates counterspeech using a pre-trained GPT-2 model.
- Saves the generated counterspeech to a new CSV file.

## Requirements
- Python 3.9.13 (Preferred)
- Virtual environment for dependency management
- Hugging Face account and API token

## Setup Instructions

### 1. Clone the Repository
```
git clone <repository-url>
cd <repository-folder>
```

### 2. Create Virtual Enviornment
```
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

### 3. Install requirements
```
pip install -r requirements.txt
```

### 4. Create a .env file in root and add Hugging Face API token
```
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
