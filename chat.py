from backend.llm.llm_factory import create_llm
from secrets.keys.get_api_key import GOOGLE_API_KEY
from backend.database.vector_database import create_vector_database, retrieve_vectors
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
import yaml


api_key = GOOGLE_API_KEY


@chain
def chatbot(values):
    prompt = template.invoke(values)
    response = model.invoke(prompt)
    return response



if __name__ == "__main__":

    # Create the Vector Database 
    create_vector_database(file_path="data/nietzsche/thus_spoke_zarathustra.pdf",
                           db_name="zarathustra")

    # Load the LLM Model
    model = create_llm(provider="google", api_key=api_key)

    # Load the personality traits
    with open("backend/personas/nietzsche.yaml") as stream:
        try:
            personality_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Prompt Template that uses personality prompting, context and user query
    template = ChatPromptTemplate.from_messages([
        ('system', personality_config['instructions'] + 
         "\n you can also use the following context: {context}"
         ),
        ('human', '{question}'),
        ])

    while True:
        query = input("Enter your query: ")

        # Retrieve relevant inforation
        retrieved_docs = retrieve_vectors(query=query, 
                                       db_name="zarathustra",
                                       n_results=5)
        
        retrieved_docs_to_single_text = "\n".join(retrieved_docs["documents"][0])
        print("\n")
        response = chatbot.invoke({"context": retrieved_docs_to_single_text, 
                                   "question": query})
        print(response.content[0]["text"])

        





    







