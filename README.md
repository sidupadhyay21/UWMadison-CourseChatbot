# UWMadison-CourseChatbot

How to use:
1. Download this repository and all necessary python libraries to your device
   - navigate into the project folder and install the libraries using pipreqs
   - pip install pipreqs
   - pipreqs (this command creates the requirements.txt based on what you need to download)
   - pip install -r requirements.txt
2. Create a .env file with your OpenAI key
   - nano .env (this will open a text editor in the terminal where you have to type out one line)
   - OPENAI_API_KEY = "<your key>"
3. Stay in project folder and run embeddings.py
   - python3 embeddings.py
   - this will generate an embeddings file, it will take around an hour to run but you only have to do this the first time
4. Run the streamlit app
   - streamlit run <location that you've saved the project>/UWMadison-CourseChatbot/embeddingChatbot.py

Optional:
5. The course data used for this chatbot came from https://github.com/maggardcolin/uw-madison-course-data. You can use this repository to get updated course data.
   - follow this repositories instructions to get a json file and move that file back into the course chatbot project folder.
   - run the toCSV.py file with your new data as an argument to prep the data to be used to make new embeddings
