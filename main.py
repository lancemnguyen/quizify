import os
import tempfile
import uuid
import json
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI


class DocumentProcessor:
    """
    This class encapsulates the functionality for processing uploaded PDF documents using Streamlit
    and Langchain's PyPDFLoader. It provides a method to render a file uploader widget, process the
    uploaded PDF files, extract their pages, and display the total number of pages extracted.
    """

    def __init__(self):
        self.pages = []  # List to keep track of pages from all documents

    def ingest_documents(self):
        """
        Renders a file uploader in a Streamlit app, processes uploaded PDF files,
        extracts their pages, and updates the self.pages list with the total number of pages.

        Given:
        - Handling of temporary files with unique names to avoid conflicts.

        Your Steps:
        1. Utilize the Streamlit file uploader widget to allow users to upload PDF files.
           Hint: Look into st.file_uploader() with the 'type' parameter set to 'pdf'.
        2. For each uploaded PDF file:
           a. Generate a unique identifier and append it to the original file name before saving it temporarily.
              This avoids name conflicts and maintains traceability of the file.
           b. Use Langchain's PyPDFLoader on the path of the temporary file to extract pages.
           c. Clean up by deleting the temporary file after processing.
        3. Keep track of the total number of pages extracted from all uploaded documents.

        Example for generating a unique file name with the original name preserved:
        ```
        unique_id = uuid.uuid4().hex
        temp_file_name = f"{original_name}_{unique_id}{file_extension}"
        ```
        """

        # Step 1: Render a file uploader widget. Replace 'None' with the Streamlit file uploader code.
        uploaded_files = st.file_uploader("Upload PDF",
                                          #####################################
                                          # Allow only type `pdf`
                                          type=["pdf"],
                                          # Allow multiple PDFs for ingestion
                                          accept_multiple_files=True
                                          #####################################
                                          )

        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                # Generate a unique identifier to append to the file's original name
                unique_id = uuid.uuid4().hex
                original_name, file_extension = os.path.splitext(uploaded_file.name)
                temp_file_name = f"{original_name}_{unique_id}{file_extension}"
                temp_file_path = os.path.join(tempfile.gettempdir(), temp_file_name)

                # Write the uploaded PDF to a temporary file
                with open(temp_file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

                # Step 2: Process the temporary file
                #####################################
                # Use PyPDFLoader here to load the PDF and extract pages.
                # https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf#using-pypdf
                # You will need to figure out how to use PyPDFLoader to process the temporary file.
                loader = PyPDFLoader(temp_file_path)
                pages = loader.load_and_split()

                # Step 3: Then, Add the extracted pages to the 'pages' list.
                #####################################
                for page in pages:
                    self.pages.append(page)

                # Clean up by deleting the temporary file.
                os.unlink(temp_file_path)

            # Display the total number of pages processed.
            st.write(f"Total pages processed: {len(self.pages)}")


class EmbeddingClient:
    """
    Task: Initialize the EmbeddingClient class to connect to Google Cloud's VertexAI for text embeddings.

    The EmbeddingClient class should be capable of initializing an embedding client with specific configurations
    for model name, project, and location. Your task is to implement the __init__ method based on the provided
    parameters. This setup will allow the class to utilize Google Cloud's VertexAIEmbeddings for processing text queries.

    Steps:
    1. Implement the __init__ method to accept 'model_name', 'project', and 'location' parameters.
       These parameters are crucial for setting up the connection to the VertexAIEmbeddings service.

    2. Within the __init__ method, initialize the 'self.client' attribute as an instance of VertexAIEmbeddings
       using the provided parameters. This attribute will be used to embed queries.

    Parameters:
    - model_name: A string representing the name of the model to use for embeddings.
    - project: The Google Cloud project ID where the embedding model is hosted.
    - location: The location of the Google Cloud project, such as 'us-central1'.

    Instructions:
    - Carefully initialize the 'self.client' with VertexAIEmbeddings in the __init__ method using the parameters.
    - Pay attention to how each parameter is used to configure the embedding client.

    Note: The 'embed_query' method has been provided for you. Focus on correctly initializing the class.
    """

    def __init__(self, model_name, project, location):
        # Initialize the VertexAIEmbeddings client with the given parameters
        # Read about the VertexAIEmbeddings wrapper from Langchain here
        # https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai
        self.client = VertexAIEmbeddings(
            #### YOUR CODE HERE ####
            model_name=model_name,
            project=project,
            location=location
        )

    def embed_query(self, query):
        """
        Uses the embedding client to retrieve embeddings for the given query.

        :param query: The text query to embed.
        :return: The embeddings for the query or None if the operation fails.
        """
        vectors = self.client.embed_query(query)
        return vectors

    def embed_documents(self, documents):
        """
        Retrieve embeddings for multiple documents.

        :param documents: A list of text documents to embed.
        :return: A list of embeddings for the given documents.
        """
        try:
            return self.client.embed_documents(documents)
        except AttributeError:
            print("Method embed_documents not defined for the client.")
            return None


class ChromaCollectionCreator:
    def __init__(self, processor, embed_model):
        """
        Initializes the ChromaCollectionCreator with a DocumentProcessor instance and embeddings configuration.
        :param processor: An instance of DocumentProcessor that has processed documents.
        :param embeddings_config: An embedding client for embedding documents.
        """
        self.processor = processor  # This will hold the DocumentProcessor from Task 3
        self.embed_model = embed_model  # This will hold the EmbeddingClient from Task 4
        self.db = None  # This will hold the Chroma collection

    def create_chroma_collection(self):
        """
        Task: Create a Chroma collection from the documents processed by the DocumentProcessor instance.

        Steps:
        1. Check if any documents have been processed by the DocumentProcessor instance. If not, display an error message using streamlit's error widget.

        2. Split the processed documents into text chunks suitable for embedding and indexing. Use the CharacterTextSplitter from Langchain to achieve this. You'll need to define a separator, chunk size, and chunk overlap.
        https://python.langchain.com/docs/modules/data_connection/document_transformers/

        3. Create a Chroma collection in memory with the text chunks obtained from step 2 and the embeddings model initialized in the class. Use the Chroma.from_documents method for this purpose.
        https://python.langchain.com/docs/integrations/vectorstores/chroma#use-openai-embeddings
        https://docs.trychroma.com/getting-started

        Instructions:
        - Begin by verifying that there are processed pages available. If not, inform the user that no documents are found.

        - If documents are available, proceed to split these documents into smaller text chunks. This operation prepares the documents for embedding and indexing. Look into using the CharacterTextSplitter with appropriate parameters (e.g., separator, chunk_size, chunk_overlap).

        - Next, with the prepared texts, create a new Chroma collection. This step involves using the embeddings model (self.embed_model) along with the texts to initialize the collection.

        - Finally, provide feedback to the user regarding the success or failure of the Chroma collection creation.

        Note: Ensure to replace placeholders like [Your code here] with actual implementation code as per the instructions above.
        """

        # Step 1: Check for processed documents
        if len(self.processor.pages) == 0:
            st.error("No documents found!", icon="🚨")
            return

        # Step 2: Split documents into text chunks
        # Use a TextSplitter from Langchain to split the documents into smaller text chunks
        # https://python.langchain.com/docs/modules/data_connection/document_transformers/character_text_splitter
        # [Your code here for splitting documents]
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

        texts = text_splitter.split_documents(self.processor.pages)

        if texts is not None:
            st.success(f"Successfully split pages to {len(texts)} documents!", icon="✅")

        # Step 3: Create the Chroma Collection
        # https://docs.trychroma.com/
        # Create a Chroma in-memory client using the text chunks and the embeddings model
        # [Your code here for creating Chroma collection]
        self.embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma.from_documents(texts, self.embed_model)

        if self.db:
            st.success("Successfully created Chroma Collection!", icon="✅")
        else:
            st.error("Failed to create Chroma Collection!", icon="🚨")

    def query_chroma_collection(self, query) -> Document:
        """
        Queries the created Chroma collection for documents similar to the query.
        :param query: The query string to search for in the Chroma collection.

        Returns the first matching document from the collection with similarity score.
        """
        if self.db:
            docs = self.db.similarity_search_with_relevance_scores(query)
            if docs:
                return docs[0]
            else:
                st.error("No matching documents found!", icon="🚨")
        else:
            st.error("Chroma Collection has not been created!", icon="🚨")

    # def as_retriever(self):
    #     return self.db.as_retriever()


class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        """
        Initializes the QuizGenerator with a required topic, the number of questions for the quiz,
        and an optional vectorstore for querying related information.

        :param topic: A string representing the required topic of the quiz.
        :param num_questions: An integer representing the number of questions to generate for the quiz, up to a maximum of 10.
        :param vectorstore: An optional vectorstore instance (e.g., ChromaDB) to be used for querying information related to the quiz topic.
        """
        if not topic:
            self.topic = "General Knowledge"
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions

        self.vectorstore = vectorstore
        self.llm = None
        self.question_bank = []  # Initialize the question bank to store questions
        self.system_template = """
            You are a subject matter expert on the topic: {topic}

            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"

            Your response must be in JSON format with exactly the following structure, and do not add any extra text such as "```json" at the beginning or end of the structure.
            Example format:
            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice>"}},
                    {{"key": "B", "value": "<choice>"}},
                    {{"key": "C", "value": "<choice>"}},
                    {{"key": "D", "value": "<choice>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct>"
            }}

            Context: {context}
            """

    def init_llm(self):
        """
        Initializes and configures the Large Language Model (LLM) for generating quiz questions.

        This method should handle any setup required to interact with the LLM, including authentication,
        setting up any necessary parameters, or selecting a specific model.

        :return: An instance or configuration for the LLM.
        """
        self.llm = VertexAI(
            model_name="gemini-pro",
            temperature=0.8,  # Increased for less deterministic questions
            max_output_tokens=500
        )

    def generate_question_with_vectorstore(self):
        """
        Generates a quiz question based on the topic provided using a vectorstore

        :return: A JSON object representing the generated quiz question.
        """
        if not self.llm:
            self.init_llm()
        if not self.vectorstore:
            raise ValueError("Vectorstore not provided.")

        from langchain_core.runnables import RunnablePassthrough, RunnableParallel

        # Enable a Retriever
        retriever = self.vectorstore.db.as_retriever()

        # Use the system template to create a PromptTemplate
        prompt = PromptTemplate.from_template(self.system_template)

        # RunnableParallel allows Retriever to get relevant documents
        # RunnablePassthrough allows chain.invoke to send self.topic to LLM
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "topic": RunnablePassthrough()}
        )
        # Create a chain with the Retriever, PromptTemplate, and LLM
        chain = setup_and_retrieval | prompt | self.llm

        # Invoke the chain with the topic as input
        response = chain.invoke(self.topic)
        return response

    def generate_quiz(self) -> list:
        """
        Task: Generate a list of unique quiz questions based on the specified topic and number of questions.

        This method orchestrates the quiz generation process by utilizing the `generate_question_with_vectorstore` method to generate each question and the `validate_question` method to ensure its uniqueness before adding it to the quiz.

        Steps:
            1. Initialize an empty list to store the unique quiz questions.
            2. Loop through the desired number of questions (`num_questions`), generating each question via `generate_question_with_vectorstore`.
            3. For each generated question, validate its uniqueness using `validate_question`.
            4. If the question is unique, add it to the quiz; if not, attempt to generate a new question (consider implementing a retry limit).
            5. Return the compiled list of unique quiz questions.

        Returns:
        - A list of dictionaries, where each dictionary represents a unique quiz question generated based on the topic.

        Note: This method relies on `generate_question_with_vectorstore` for question generation and `validate_question` for ensuring question uniqueness. Ensure `question_bank` is properly initialized and managed.
        """
        self.question_bank = []  # Reset the question bank

        # for _ in range(self.num_questions):
        while len(self.question_bank) < self.num_questions:
            ##### YOUR CODE HERE #####
            question_str = QuizGenerator.generate_question_with_vectorstore(
                self)  # Use class method to generate question

            ##### YOUR CODE HERE #####
            try:
                # Convert the JSON String to a dictionary
                question_dict = json.loads(question_str)
            except json.JSONDecodeError:
                print("Failed to decode question JSON.")
                # print(question_str)
                continue  # Skip this iteration if JSON decoding fails
            ##### YOUR CODE HERE #####

            ##### YOUR CODE HERE #####
            # Validate the question using the validate_question method
            if self.validate_question(question_dict):
                print("Successfully generated unique question")
                # print(question_str)
                # Add the valid and unique question to the bank
                self.question_bank.append(question_dict)
            else:
                print("Duplicate or invalid question detected.")
            ##### YOUR CODE HERE #####

        return self.question_bank

    def llm_judge(self, question_dict):
        prompt_template = """
        Check if the current question has the same meaning as the previous questions. If it does, just return False, otherwise just return True. Don't add any explanations.
        Question:
        {question}

        Previous questions:
        {prev_questions}
        """
        question = question_dict['question']
        prev_questions = ""
        for q in self.question_bank:
            prev_questions += q['question'] + "\n"
        # print(prev_questions)
        prompt_template = prompt_template.format(question=question, prev_questions=prev_questions)
        # print(prompt_template)
        judge_prompt = PromptTemplate.from_template(prompt_template)

        chain = judge_prompt | self.llm

        # Invoke the chain with the topic as input
        response = chain.invoke({"question": question})
        # print(response)
        return response

    def validate_question(self, question: dict) -> bool:
        """
        Task: Validate a quiz question for uniqueness within the generated quiz.

        This method checks if the provided question (as a dictionary) is unique based on its text content compared to previously generated questions stored in `question_bank`. The goal is to ensure that no duplicate questions are added to the quiz.

        Steps:
            1. Extract the question text from the provided dictionary.
            2. Iterate over the existing questions in `question_bank` and compare their texts to the current question's text.
            3. If a duplicate is found, return False to indicate the question is not unique.
            4. If no duplicates are found, return True, indicating the question is unique and can be added to the quiz.

        Parameters:
        - question: A dictionary representing the generated quiz question, expected to contain at least a "question" key.

        Returns:
        - A boolean value: True if the question is unique, False otherwise.

        Note: This method assumes `question` is a valid dictionary and `question_bank` has been properly initialized.
        """
        ##### YOUR CODE HERE #####
        # Consider missing 'question' key as invalid in the dict object
        # Check if a question with the same text already exists in the self.question_bank
        if len(self.question_bank) == 0:
            return True
        result = self.llm_judge(question)
        if result.startswith("True"):
            return True
        else:
            # print(question)
            return False
        ##### YOUR CODE HERE #####
        # return is_unique


class QuizManager:
    ##########################################################
    def __init__(self, questions: list):
        """
        Task: Initialize the QuizManager class with a list of quiz questions.

        Overview:
        This task involves setting up the `QuizManager` class by initializing it with a list of quiz question objects. Each quiz question object is a dictionary that includes the question text, multiple choice options, the correct answer, and an explanation. The initialization process should prepare the class for managing these quiz questions, including tracking the total number of questions.

        Instructions:
        1. Store the provided list of quiz question objects in an instance variable named `questions`.
        2. Calculate and store the total number of questions in the list in an instance variable named `total_questions`.

        Parameters:
        - questions: A list of dictionaries, where each dictionary represents a quiz question along with its choices, correct answer, and an explanation.

        Note: This initialization method is crucial for setting the foundation of the `QuizManager` class, enabling it to manage the quiz questions effectively. The class will rely on this setup to perform operations such as retrieving specific questions by index and navigating through the quiz.
        """
        ##### YOUR CODE HERE #####
        self.questions = questions
        self.total_questions = len(self.questions)

    ##########################################################

    def get_question_at_index(self, index: int):
        """
        Retrieves the quiz question object at the specified index. If the index is out of bounds,
        it restarts from the beginning index.

        :param index: The index of the question to retrieve.
        :return: The quiz question object at the specified index, with indexing wrapping around if out of bounds.
        """
        # Ensure index is always within bounds using modulo arithmetic
        valid_index = index % self.total_questions
        return self.questions[valid_index]

    ##########################################################
    def next_question_index(self, direction=1):
        """
        Task: Adjust the current quiz question index based on the specified direction.

        Overview:
        Develop a method to navigate to the next or previous quiz question by adjusting the `question_index` in Streamlit's session state. This method should account for wrapping, meaning if advancing past the last question or moving before the first question, it should continue from the opposite end.

        Instructions:
        1. Retrieve the current question index from Streamlit's session state.
        2. Adjust the index based on the provided `direction` (1 for next, -1 for previous), using modulo arithmetic to wrap around the total number of questions.
        3. Update the `question_index` in Streamlit's session state with the new, valid index.
            # st.session_state["question_index"] = new_index

        Parameters:
        - direction: An integer indicating the direction to move in the quiz questions list (1 for next, -1 for previous).

        Note: Ensure that `st.session_state["question_index"]` is initialized before calling this method. This navigation method enhances the user experience by providing fluid access to quiz questions.
        """
        ##### YOUR CODE HERE #####
        current_index = st.session_state["question_index"]
        new_index = (current_index + direction) % self.total_questions
        st.session_state["question_index"] = new_index
    ##########################################################


if __name__ == "__main__":

    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "gemini-quizify-420521",
        "location": "us-central1"
    }

    # Add Session State
    if 'question_bank' not in st.session_state or len(st.session_state['question_bank']) == 0:

        ##### YOUR CODE HERE #####
        # Step 1: init the question bank list in st.session_state
        st.session_state["question_bank"] = []
        ##### YOUR CODE HERE #####

        screen = st.empty()
        with screen.container():
            st.header("Quiz Builder")

            # Create a new st.form flow control for Data Ingestion
            with st.form("Load Data to Chroma"):
                st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")

                processor = DocumentProcessor()
                processor.ingest_documents()

                embed_client = EmbeddingClient(**embed_config)

                chroma_creator = ChromaCollectionCreator(processor, embed_client)

                ##### YOUR CODE HERE #####
                # Step 2: Set topic input and number of questions
                topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
                questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
                ##### YOUR CODE HERE #####

                submitted = st.form_submit_button("Submit")

                if submitted:
                    chroma_creator.create_chroma_collection()
                    if len(processor.pages) > 0:
                        st.write(f"Generating {questions} questions for topic: {topic_input}")

                    ##### YOUR CODE HERE #####
                    generator = QuizGenerator(topic_input, questions,
                                              chroma_creator)  # Step 3: Initialize a QuizGenerator class using the topic, number of questions, and the chroma collection
                    question_bank = generator.generate_quiz()
                    # Step 4: Initialize the question bank list in st.session_state
                    st.session_state["question_bank"] = question_bank
                    # Step 5: Set a display_quiz flag in st.session_state to True
                    st.session_state["display_quiz"] = True
                    # Step 6: Set the question_index to 0 in st.session_state
                    st.session_state["question_index"] = 0
                    st.rerun()
                    ##### YOUR CODE HERE #####

    elif st.session_state["display_quiz"]:

        st.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            question_bank = st.session_state["question_bank"]
            quiz_manager = QuizManager(question_bank)

            # Format the question and display it
            with st.form("MCQ"):
                ##### YOUR CODE HERE #####
                # Step 7: Set index_question using the Quiz Manager method get_question_at_index passing the st.session_state["question_index"]
                index_question = quiz_manager.get_question_at_index(st.session_state["question_index"])
                ##### YOUR CODE HERE #####

                # Unpack choices for radio button
                choices = []
                for choice in index_question['choices']:
                    key = choice['key']
                    value = choice['value']
                    choices.append(f"{key}) {value}")

                # Display the Question
                st.write(f"{st.session_state['question_index'] + 1}. {index_question['question']}")
                answer = st.radio(
                    "Choose an answer",
                    choices,
                    index=None
                )

                answer_choice = st.form_submit_button("Submit")

                ##### YOUR CODE HERE #####
                # Step 8: Use the example below to navigate to the next and previous questions
                # Here we use the next_question_index method from our quiz_manager class
                st.form_submit_button("Next Question", on_click=lambda: quiz_manager.next_question_index(direction=1))
                st.form_submit_button("Previous Question",
                                      on_click=lambda: quiz_manager.next_question_index(direction=-1))
                ##### YOUR CODE HERE #####

                if answer_choice and answer is not None:
                    correct_answer_key = index_question['answer']
                    if answer.startswith(correct_answer_key):
                        st.success("Correct!")
                    else:
                        st.error("Incorrect!")
                    st.write(f"Explanation: {index_question['explanation']}")
