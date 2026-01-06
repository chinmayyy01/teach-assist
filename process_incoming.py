from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
from google import genai 
from config import api_key

client = genai.Client(api_key=api_key)

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

# When you wanna use llama model locally use this function
# def inference(prompt):
#     r = requests.post("http://localhost:11434/api/generate", json={
#         "model": "llama3.2",
#         "prompt": prompt,
#         "stream": False,
#     })
    
#     response = r.json()
#     print(response)
    
#     return response

#Using Gemini model from Google Cloud
def inference(prompt):
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return {"response": response.text}
    except Exception as e:
        return {"response": f"An error occurred: {str(e)}"}

df = joblib.load('embeddings.joblib')

incoming_query = input("Enter your query: ")
question_embedding = create_embedding([incoming_query])[0]

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
# print(new_df[['title', 'number', 'text']])

prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
________________________________________________________________________________________
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''
with open('prompt.txt', 'w') as f:
    f.write(prompt)

# response = inference(prompt)["response"]
# print(response)
# This part of your code is now correct, provided the function above is updated
result_data = inference(prompt)
response = result_data["response"]
print(response)

with open('response.txt', 'w', encoding='utf-8') as f:
    f.write(response)
# for index, item in new_df.iterrows():
#     print(index, item['title'], item['number'], item['text'], item['start'], item['end'])