import speech_recognition as sr
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

engine= sr.Recognizer()
mic= sr.Microphone()
hasil=""

with mic as source:
    print("Ucapkan sesuatu")
    rekaman= engine.listen(source, phrase_time_limit=15)

    try:
        hasil=engine.recognize_google(rekaman,language='id-ID')
        print(hasil)
sentences = [
            '',
            'jenis filter yang digunakan dalam pemrosesan sinyal audio dan digunakan untuk mengspektralubah karakteristik  dari suara agar lebih sesuai dengan persepsi manusia terhadap suara tersebut'
        ]
vectorizer = CountVectorizer().fit_transform(sentences)
vectors = vectorizer.toarray()
        # print(vectors)
csim = cosine_similarity(vectors)
        # print(csim)
        
def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1, vec2) [0][0]
cosine = cosine_sim_vectors(vectors[0], vectors[1])
print('score :',cosine)
# except engine.UnknowValueEror:
# print("Mohon maaf tidak dapat dideteksi")
#     except Exception as e:
# print(e)
