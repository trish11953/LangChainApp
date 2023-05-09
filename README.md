
# FAQ Matcher
This is a Python script that uses LangChain embeddings, Faiss, and a PyQt5 graphical user interface (GUI) to match a query with the most relevant question-answer pair from a PDF file or files of frequently asked questions (FAQs). It also gives suggestion questions that the user might want to know the answer to.

## Installation
Clone this repository:

```
git clone https://github.com/trish11953/LangChainApp.git
```
Install the required packages:

```
pip install -r requirements.txt
```
Note: If you are using a GPU and want to use the GPU version of Faiss, replace `faiss-cpu` with `faiss-gpu` in the command above.

Additionally, install 'en_core_web_sm' model for SpaCy using this command in your terminal.
```
python -m spacy download en_core_web_sm
```
Set your OpenAI API key as an environment variable in your terminal:

For Linux and macOS:
```
export OPENAI_API_KEY=<your-openai-api-key>
```
For Windows:
```
set OPENAI_API_KEY=<your-openai-api-key>
```

## Usage
To use the FAQ matcher with the PyQt5 GUI, run the following command in your terminal:

```
python faq.py
```
This will launch the GUI application. Browse for a PDF file or files containing FAQs, load the files, enter a query in the input field, and click "Submit" to process the query and display the matched or generated answer. It will also give related questions.
