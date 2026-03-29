from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# загружаем модель
model = SentenceTransformer('all-MiniLM-L6-v2')

# тестовые тексты нормативных норм
texts = [
    "Гражданин обязан платить налог",
    "Гражданин должен уплачивать налог",
    "Гражданину запрещено уклоняться от налога"
]

# делаем вектора
embeddings = model.encode(texts)

# считаем сходство
sim = cosine_similarity(embeddings)

# сравниваем тексты
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        print("------")
        print("Текст 1:", texts[i])
        print("Текст 2:", texts[j])
        print("Сходство:", round(sim[i][j], 2))

        if sim[i][j] > 0.85:
            print("Вывод: возможный дубль")
            print("Причина: высокая смысловая близость")

        elif sim[i][j] > 0.7:
            print("Вывод: похожие нормы")
            print("Причина: общий контекст, но возможны различия")

            if ("запрещено" in texts[i].lower() and "обязан" in texts[j].lower()) or \
               ("запрещено" in texts[j].lower() and "обязан" in texts[i].lower()):
                print("Дополнительно: возможное противоречие")

        else:
            print("Вывод: тексты различаются")