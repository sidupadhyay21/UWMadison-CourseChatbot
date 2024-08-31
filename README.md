# UWMadison-CourseChatbot

How to use:
1. Download this repository and all necessary python libraries to your device
   - You can install the libraries using pipreqs
   - pip install pipreqs
   - pipreqs (this command creates the requirements.txt based on what you need to download)
   - pip install -r requirements.txt
2. Navigate into the project folder and run embeddings.py
   - python3 embeddings.py
   - this will generate an embeddings file, it will take around an hour to run but you only have to do this the first time
3. Run the streamlit app
   - streamlit run <location that you've saved the project>/UWMadison-CourseChatbot/embeddingChatbot.py
