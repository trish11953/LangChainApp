import sys
import os
import faiss
import numpy as np
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import pdfplumber
import re
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QFileDialog, \
    QPlainTextEdit
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages])

    # Splits the text into lines
    lines = text.split("\n")

    # Uses regex patterns to identify questions and answers
    question_pattern = re.compile(r'^\s*(?:Where|How|Is|When|Can|What|Do|Are|Which|Will|Why).+\?$')

    # Creates an empty dictionary to store questions and answers
    q_and_a = {}
    current_question = ''
    current_answer = ''
    for line in lines:
        if question_pattern.match(line):
            # If a new question is detected, adds the current question and answer to the dictionary
            if current_question and current_answer:
                q_and_a[current_question] = current_answer.strip()
                current_answer = ''
            current_question = line.strip()
        elif current_question:
            # If there is a current question, adds the current line to the current answer
            current_answer += line.strip() + ' '

    # Adds the final question and answer to the dictionary
    if current_question and current_answer:
        q_and_a[current_question] = current_answer.strip()

    # Separates the questions and answers into separate lists
    questions = list(q_and_a.keys())
    answers = list(q_and_a.values())

    return questions, answers


def train_index(questions, embeddings):
    dimension = len(embeddings[0])  # Gets the dimensionality of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Creates a new index of the given dimension
    index.add(np.array(embeddings).astype("float32"))  # Adds the embeddings to the index
    return index


def match_query(index, query_embedding, questions, answers, threshold=0.8):
    # Searches the index for the nearest neighbor to the query embedding
    D, I = index.search(np.array([query_embedding]).astype("float32"), 1)
    # Gets the index of the best matching question-answer pair
    best_match_index = I[0][0]
    # Calculates the similarity score of the best match
    best_match_score = 1 - D[0][0]
    # If the similarity score is below the threshold, returns no match
    if best_match_score < threshold:
        return None, None
    else:
        # Otherwise, returns the best matching question and answer strings
        return questions[best_match_index], answers[best_match_index]


def get_suggestions(index, query_embedding, questions, answers, num_suggestions=3):
    # Searches the index for the nearest neighbors to the query embedding
    D, I = index.search(np.array([query_embedding]).astype("float32"), num_suggestions + 1)

    suggestions = []
    for i in range(num_suggestions):
        suggestions.append(questions[I[0][i + 1]])

    return suggestions


def get_gpt3_suggestions(query, num_suggestions=3):
    # Creates a prompt string to ask GPT-3 to generate related questions
    prompt = f"Generate {num_suggestions} related questions to \"{query}\":"
    for i in range(1, num_suggestions + 1):
        prompt += f"\n{i}."
    # Calls the GPT-3 API with the created prompt
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    # Extracts the suggestions from the API response
    suggestions = []
    raw_suggestions = response.choices[0].text.strip().split('\n')[1:]
    for suggestion in raw_suggestions:
        # Cleans up the suggestion text by removing the suggestion number and extra spaces
        suggestions.append(suggestion.strip().lstrip("123."))

    return suggestions


class FileDropTextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)  # Enables drop events for the widget

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():  # Checks if the dropped item is a URL
            event.accept()  # Accepts the event if it is a URL
        else:
            event.ignore()  # Otherwise, ignores the event

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():  # Checks if the dropped item is a URL
            event.setDropAction(Qt.CopyAction)  # Sets the action to Copy
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            pdf_files = []
            for url in event.mimeData().urls():  # Loops over the URLs
                file_path = url.toLocalFile()  # Gets the local file path from the URL
                if file_path.endswith(".pdf"):  # Checks if the file is a PDF file
                    pdf_files.append(file_path)  # Adds the file path to the list of PDF files
            self.setPlainText("\n".join(pdf_files))  # Sets the plain text of the widget to the list of PDF file paths
        else:
            event.ignore()


class FaqMatcherApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.files_data = []  # New variable to store data for multiple documents
        self.openai_embeddings = OpenAIEmbeddings()  # Creates openai_embeddings object here

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)

        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)

        self.setWindowTitle('FAQ Matcher')

        self.file_label = QLabel('FAQ PDF File:')
        self.file_label.setFont(title_font)
        layout.addWidget(self.file_label)

        # Adds an instructional label
        self.instructions_label = QLabel('Drag and drop upto 10 PDF files or click "Browse" to select:')
        layout.addWidget(self.instructions_label)

        self.file_input = FileDropTextEdit()
        self.file_input.setFixedHeight(60)  # Sets the desired height for file_input
        layout.addWidget(self.file_input)

        self.browse_button = QPushButton('Browse')
        self.browse_button.clicked.connect(self.browse_pdf)
        layout.addWidget(self.browse_button)

        self.load_files_button = QPushButton('Upload Files')
        self.load_files_button.clicked.connect(self.load_files)
        layout.addWidget(self.load_files_button)

        self.load_text = QPlainTextEdit()
        self.load_text.setFixedHeight(30)  # Sets the desired height here
        self.load_text.setReadOnly(True)
        layout.addWidget(self.load_text)

        self.query_label = QLabel('Query:')
        self.query_label.setFont(title_font)
        layout.addWidget(self.query_label)

        self.query_input = QLineEdit()
        layout.addWidget(self.query_input)

        self.submit_button = QPushButton('Submit')
        self.submit_button.clicked.connect(self.process_query)
        layout.addWidget(self.submit_button)

        self.answer_label = QLabel('Answer:')
        self.answer_label.setFont(title_font)
        layout.addWidget(self.answer_label)

        self.result_text = QTextEdit()
        self.result_text.setFixedHeight(150)  # Sets the desired height here
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.suggestion_label = QLabel("Related questions:")
        self.suggestion_label.setFont(title_font)
        layout.addWidget(self.suggestion_label)

        self.suggestion_text = QTextEdit()
        self.suggestion_text.setReadOnly(True)
        layout.addWidget(self.suggestion_text)

        self.setLayout(layout)
        self.setFixedSize(600, 800)  # Adjusts the height of the main window accordingly
        self.apply_stylesheet()

    def apply_stylesheet(self):
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(54, 57, 63))
        self.setPalette(p)

        stylesheet = """
        QLabel {
            color: #F1F1F1;
        }
        QLineEdit {
            background-color: #2E3136;
            color: #F1F1F1;
            border: 1px solid #3D4146;
            border-radius: 5px;
            padding: 5px;
        }
        QPushButton {
            background-color: #7289DA;
            color: #F1F1F1;
            border: none;
            border-radius: 5px;
            padding: 5px 15px;
        }
        QPushButton:hover {
            background-color: #8794DE;
        }
        QTextEdit {
            background-color: #2E3136;
            color: #F1F1F1;
            border: 1px solid #3D4146;
            border-radius: 5px;
            padding: 5px;
        }
        QPlainTextEdit {
            background-color: #2E3136;
            color: #559900;
            border: 1px solid #3D4146;
            border-radius: 5px;
            padding: 5px;
        }
        """
        self.setStyleSheet(stylesheet)

    def browse_pdf(self):
        # Configures options for the file dialog
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        # Opens a file dialog to allow the user to select one or more PDF files
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select PDF Files", "", "PDF Files (*.pdf)", options=options)
        # If one or more files were selected, updates the text in the input box to show the selected file paths
        if file_names:
            self.file_input.setPlainText("\n".join(file_names))

    def load_files(self):
        pdf_files = self.file_input.toPlainText().split("\n")[:10]  # Limits to 10 files

        if not pdf_files:
            self.result_text.setText("Please provide at least one PDF file.")
            return

        combined_qa = []

        for pdf_file in pdf_files:
            # Extracts FAQs from PDF
            questions, answers = extract_text_from_pdf(pdf_file)

            # Combines questions and answers from all documents
            combined_qa.extend(zip(questions, answers))

        # Gets embeddings for combined questions using self.openai_embeddings
        combined_question_embeddings = [self.openai_embeddings.embed_documents([qa[0]])[0] for qa in combined_qa]

        # Stores the data for the combined documents
        self.files_data = [{"qa": combined_qa, "embeddings": combined_question_embeddings}]

        self.load_text.setPlainText("Files uploaded successfully.")

    def process_query(self):
        query = self.query_input.text()

        if not self.files_data:
            self.result_text.setText("Please load PDF files first.")
            return

        if not query:
            self.result_text.setText("Please provide a query.")
            return

        # Sets up OpenAI API key for Langchain and GPT-3
        openai.api_key = os.environ.get("OPENAI_API_KEY")

        # Gets the query embedding using self.openai_embeddings
        query_embedding = self.openai_embeddings.embed_query(query)

        # Extracts combined questions, answers, and embeddings from self.files_data
        combined_qa = self.files_data[0]["qa"]
        combined_question_embeddings = self.files_data[0]["embeddings"]

        # Trains the index and match the query
        index = train_index([qa[0] for qa in combined_qa], combined_question_embeddings)
        matched_question, matched_answer = match_query(index, query_embedding, [qa[0] for qa in combined_qa],
                                                       [qa[1] for qa in combined_qa])

        if matched_question:
            # If a match was found in any PDF file, sets the result_text content
            self.result_text.setText(f"{matched_answer} \n\nAnswer was matched from PDF.")
            suggestions = get_suggestions(index, query_embedding, [qa[0] for qa in combined_qa],
                                          [qa[1] for qa in combined_qa])
        else:
            # If no match was found, uses GPT-3 to generate a response
            prompt = f"{query}?"
            response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=1024)
            generated_answer = response.choices[0].text.strip()

            self.result_text.setText(f"{generated_answer} \n\nAnswer was generated.")
            suggestions = get_gpt3_suggestions(query)

        suggestion_text = "\n".join(suggestions)
        self.suggestion_text.setText(suggestion_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FaqMatcherApp()
    ex.show()
    sys.exit(app.exec_())
