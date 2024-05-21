# Установка библиотек
'''
pip install natasha
'''


# Подключение библиотек
import os                # Управление операционной системой
import pandas as pd      # Data frames
import re                # Разбиение текста на слова
import numpy as np       # Список в виде массива

from sklearn.feature_extraction.text import TfidfVectorizer # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py




#
# Отпределение путей ко всем файлам с исходными данными
#



# Пути основные
cwd = os.getcwd()               # Текущая директория
cwd_data = cwd + '\Data\\'      # Путь до директории с исходными данными

# Пути дополнительные
filePath_stopWords = cwd + '\\' + 'stop-words.txt'  # Путь к файлу со стоп-словами, скаченному по ссылке: https://snipp.ru/seo/stop-ru-words

# Пути к папкам с рубриками
foldersNames_list = os.listdir(cwd_data)  # Список папок с рубриками
rubricsCount = len(foldersNames_list)     # Число рубрик
foldersPaths_list = []                    # Список путей к папкам с рубриками
for i in range(rubricsCount):
    foldersPaths_list.append(cwd_data + foldersNames_list[i])




# Функция получения путей ко всем статьям в папке рубрики 
def getFilesPathsInRubricFolder(folderPath_rubric):
    """
    folderPath_rubric - Путь к папке рубрики
    """
    # Файлы в папке рубрики
    filesNames_rubric_list = os.listdir(folderPath_rubric)  # Названия файлов со статьями в папке рубрики
    filesCount = len(filesNames_rubric_list)                # Число файлов (статей) в папке рубрики
    
    # Пути ко всем статьям в папке рубрики
    filesPaths_rubric_list = [] # Список путей ко всем статьям в папке рубрики
    for i in range(filesCount):
        filesPaths_rubric_list.append(folderPath_rubric + '\\' + filesNames_rubric_list[i])
    
    return filesPaths_rubric_list
    
    
    
    
# Пути ко всем статьям в папках рубрик
foldersPaths_all = []             # Пути ко всем статьям в папках рубрик
foldersPaths_all_rubrics = []     # Рубрика, соответствующая пути к статье (для создания таблицы Рубрика:Путь_к_статье)
for i in range(rubricsCount):
    
    
    foldersPaths_inRubric = getFilesPathsInRubricFolder(folderPath_rubric = foldersPaths_list[i])
    foldersPaths_all += foldersPaths_inRubric
    
    arcticlesCount_inRubric = len(foldersPaths_inRubric)
    
    
    for j in range(arcticlesCount_inRubric):
        foldersPaths_all_rubrics.append(foldersNames_list[i])

# Таблица [Рубрика, Путь_к_статье]
df_rubricsAndArticlesPaths_all = pd.DataFrame({'Rubric' : foldersPaths_all_rubrics,
                                          'ArticlePath' : foldersPaths_all})

articlesCount = len(foldersPaths_all) # Число статей всего




#
# Пути к подготовленным текстам
#

# Путь к папке с подготовленными текстами
folderPath_preparedData = cwd + '\\' + 'Data_prepared' 

# Пути к папкам рубрик
folderPath_preparedData_folders = os.listdir(folderPath_preparedData) # Список папок
foldersPaths_preparedData_rubrics = [] # Пути к папкам рубрикам с подготовленными текстами
for i in range(rubricsCount):
    path = folderPath_preparedData + '\\' + foldersNames_list[i]
    foldersPaths_preparedData_rubrics.append(path)

# Пути ко всем статьям в папках рубрик
foldersPaths_preparedData_all = []             # Пути ко всем статьям в папках рубрик (подготовленные тексты)
for i in range(articlesCount):
    path = foldersPaths_all[i].replace('TextClassifier\\Data', 'TextClassifier\\Data_prepared', 1)
    foldersPaths_preparedData_all.append(path)



#-----------------------------------------------------------------
#                   Часть 1. Подготовка текстов
#-----------------------------------------------------------------




#
# Импорт исходных данных
#



# Функция считывания текста из файла txt (+ удаление пустых строк, + простая очистка)
def readTextFromFile(filePath):

    # Считывание текста из файла построчно
    with open(filePath, 'r', encoding = 'utf8') as f:
        contents_list = f.readlines()      # Строки в файле
    
    # Объединение строк в единый текст и очистка
    text = ''                              # Текст из файла
    for i in range(len(contents_list)):
        text += contents_list[i]           # Объединение
        
        # Очистка от знаков препинания
        text = text.replace('.', '')      
        text = text.replace(',', '') 
        text = text.replace(' — ', ' ')
        text = text.replace(' - ', ' ')
        
        text = text.replace('\n', ' ')      # Очистка от символа \n
        text = text.replace('  ', ' ')      # Очистка от двойных пробелов
        

    return text
    
   
# Список текстов
articlesTexts_list = []
for i in range(articlesCount):
    filePath = df_rubricsAndArticlesPaths_all.iloc[i, 1]
    text = readTextFromFile(filePath = filePath)  
    articlesTexts_list.append(text)




#
# Подготовка текстов
#



# 1. Удаление стоп-слов

stopWords_list = [] # Список стоп-слов

# Считывание текста из файла со стоп-словами построчно
with open(filePath_stopWords, 'r', encoding = 'utf8') as f:
    stopWords_list = f.readlines()      # Строки в файле
# Очистка от символа \n
stopWords_list_len = len(stopWords_list) # Число стоп-слов
for i in range(stopWords_list_len):
    stopWords_list[i] = stopWords_list[i].replace('\n', '')

# Удаление стоп-слов
text_words_list = re.findall(r'\b\S+\b', articlesTexts_list[0]) # Список всех слов из текста (r'\b\S+\b' - регулярное (шаблонное) выражение, где \b - граница слова, \S - непробельный знак, \S+ - любая последовательность \S )                  
text_words_list_withoutStopWords = [] # Список всех слов из текста, кроме стоп-слов
wordsCount_inText = len(text_words_list)
for i in range(wordsCount_inText):
    
    word = text_words_list[i].lower() # Слово (.lower() - с маленькой буквы)
    
    # Если слово НЕ входит в список стоп-слов, то сохранение в список text_words_list_withoutStopWords
    if not word in stopWords_list:
        text_words_list_withoutStopWords.append(text_words_list[i])



# 2. удалить именованные сущности и цифры (библиотека natasha)
# https://habr.com/ru/post/516098/

# Подключение элементов библиотеки natasha
from natasha import (
    Segmenter,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    
    Doc
)
from natasha import NewsNERTagger
from natasha import MorphVocab
morph_vocab = MorphVocab()



# Функция полученя подготовленных текстов (выполнение 3-х видов очистки)
def getPreparedTexts(textInitial):
    """
    Parameters:
    textInitial - Исходный текст
    
    Returns:
    text_prepared_1 - Подготовленный текст после 1 вида очитски;
    text_prepared_2 - Подготовленный текст после 2 вида очитски;
    text_prepared_3 - Подготовленный текст после 3 вида очитски.

    """
    
    
    textInitial_words_list = re.findall(r'\b\S+\b', textInitial) # Список всех слов в тексте исходном (r'\b\S+\b' - регулярное (шаблонное) выражение, где \b - граница слова, \S - непробельный знак, \S+ - любая последовательность \S )                  
    textInitial_wordsCount = len(textInitial_words_list)         # Число слов в тексте исходном
    
    
    
    # 1. Удаление стоп-слов
    text_prepared_1 = []                         # Подготовленный текст после 1 вида очистки
    textInitial_words_list_withoutStopWords = [] # Список всех слов из текста, кроме стоп-слов
    for i in range(textInitial_wordsCount):
        word = textInitial_words_list[i].lower()        # Слово (.lower() - с маленькой буквы)
        # Если слово НЕ входит в список стоп-слов, то сохранение в список text_words_list_withoutStopWords
        if not word in stopWords_list:
            textInitial_words_list_withoutStopWords.append(textInitial_words_list[i])
    text_prepared_1_list = textInitial_words_list_withoutStopWords
    # Представлние списка слов в виде единого текста
    text_prepared_1 = ' '.join(word for word in text_prepared_1_list)
    
    

    # 2. Удаление именованных сущностей и цифр (библиотека natasha)
    # https://habr.com/ru/post/516098/
    text_prepared_2 = []                         # Подготовленный текст после 1 вида очистки
    
    # 2.1 Удаление именованных сущностей
    text_prepared_2_1 = []
    
    segmenter = Segmenter()

    emb = NewsEmbedding()
    
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    
    
    
    #text = 'Посол Израиля на Украине Йоэль Лион признался, что пришел в шок, узнав о решении властей Львовской области объявить 2019 год годом лидера запрещенной в России Организации украинских националистов (ОУН) Степана Бандеры...'
    text = text_prepared_1
    doc = Doc(text)
    
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    
    
    #from natasha import NewsNERTagger
    ner_tagger = NewsNERTagger(emb)
    doc.tag_ner(ner_tagger)
    #doc.ner.print()
    
    
    
    #from natasha import MorphVocab
    morph_vocab = MorphVocab()
    
    
    # Поиск именованных сущностей
    namedEntities_list = []               # Именованные сущности в тексте
    namedEntities_count = len(doc.spans)  # Число именованных сущностей в тексте
    for i in range(namedEntities_count):
        namedEntities_list.append(doc.spans[i].text)
    
    # Удаление именованных сущностей
    text1 = text
    for i in range(namedEntities_count):
        text1 = text1.replace(namedEntities_list[i], '')
    
    # Удаление лишних дйоных и тройных пробелов
    text1 = text1.replace('   ', ' ')
    text1 = text1.replace('  ', ' ')
    
    
    text_prepared_2_1 = text1
    
    
    
    # 2.2 Поиск и удаление цифр
    text_prepared_2_1_words_list = re.findall(r'\b\S+\b', text_prepared_2_1) # Список всех слов в тексте исходном (r'\b\S+\b' - регулярное (шаблонное) выражение, где \b - граница слова, \S - непробельный знак, \S+ - любая последовательность \S )                  
    text_prepared_2_1_wordsCount = len(text_prepared_2_1_words_list)         # Число слов в тексте исходном
    text_words_list_withoutNumbers = [] # Список всех слов из текста, кроме цифр
    for i in range(text_prepared_2_1_wordsCount):
        
        word = text_prepared_2_1_words_list[i] # Слово 
        
        
        
        # Если слово НЕ равно числу, то сохранение в список text_words_list_withoutNumbers
        if word.isdigit() != True:
            text_words_list_withoutNumbers.append(word)

    text_prepared_2_list = text_words_list_withoutNumbers
    # Представлние списка слов в виде единого текста
    text_prepared_2 = ' '.join(word for word in text_prepared_2_list)

    


    
    # 3. Лемматизация (лемма - начальная форма слова)
    # от каждого слова взять только стемму/лемму (библиотека pymorphy)
    #https://www.youtube.com/watch?v=Tyk0aHHzlYE
    text_prepared_3 = []                         # Подготовленный текст после 3 вида очистки
    
    # Подготовка текста к библиотеке natasha
    doc = Doc(text_prepared_2)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    # Лемматизация (определение леммы для каждого слова)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    dict_wordAndLemma = {_.text: _.lemma for _ in doc.tokens}   # Словарь [Слово:Лемма]
    text_prepared_3_list = list(dict_wordAndLemma.values())     # Список лемм
    # Представлние списка слов (лемм) в виде единого текста
    text_prepared_3 = ' '.join(word for word in text_prepared_3_list)
  
    
    return text_prepared_1, text_prepared_2, text_prepared_3 # , text_prepared_2_1
    

# Получение подготовленных текстов (выполнение 3-х видов очистки)
texts_prepared_1 = [] # Подготовленные тексты после 1 вида очистки
texts_prepared_2 = [] # Подготовленные тексты после 2 вида очистки
texts_prepared_3 = [] # Подготовленные тексты после 3 вида очистки
for i in range(articlesCount):
    prepText_1, prepText_2, prepText_3 = getPreparedTexts(textInitial = articlesTexts_list[i])
    texts_prepared_1.append(prepText_1)
    texts_prepared_2.append(prepText_2)
    texts_prepared_3.append(prepText_3)
    print(i)


# Таблица [Рубрика, Текст_статьи, Текст_очищенный_1, Текст_очищенный_2, Текст_очищенный_3]
df_main = pd.DataFrame({'Rubric' : foldersPaths_all_rubrics,
                        'Text' : articlesTexts_list,
                        'Text_prep_1' : texts_prepared_1,
                        'Text_prep_2' : texts_prepared_2,
                        'Text_prep_3' : texts_prepared_3})

  

#
# Сохранение подготовленных текстов
#


# Создание папки, если не существует
folderPath_preparedData = cwd + '\\' + 'Data_prepared' # Путь к папке с подготовленными текстами
if not 'Data_prepared' in os.listdir(cwd):
    os.mkdir(folderPath_preparedData) # Создание папки

# Создание папок рубрик, если не существуют
for i in range(rubricsCount):
    if not foldersNames_list[i] in folderPath_preparedData_folders:
        os.mkdir(foldersPaths_preparedData_rubrics[i]) # Создание папки
  
# Сохранение подготовленных текстов
for i in range(articlesCount):
    with open(foldersPaths_preparedData_all[i], 'w', encoding = 'utf8') as text_file:
        text_file.write(texts_prepared_3[i])






#-----------------------------------------------------------------
#                   Часть 2. Модель KNN
#-----------------------------------------------------------------



#
# Импорт подготовленных текстов
#



# Список текстов подготовленных
articlesTexts_prepared_list = []
for i in range(articlesCount):
    with open(foldersPaths_preparedData_all[i], 'r', encoding = 'utf8') as text_file:
        contents_list = text_file.readlines()               # Текст в файле (построчно)
        articlesTexts_prepared_list.append(contents_list[0])   # Сохранение текста в список

# Список имен файлов с текстами
articlesTexts_prepared_filesNames = []
for i in range(articlesCount):
    fileName = foldersPaths_preparedData_all[i]
    fileName = fileName[fileName.rfind('\\') + 1 :]
    articlesTexts_prepared_filesNames.append(fileName)


# Номера рубрик
rubricsNumbers_list = []        # Норера рубрик для всех статей (n = числу статей)
for i in range(articlesCount):
    rubricName = df_rubricsAndArticlesPaths_all.iloc[i, 0]
    # Поиск номера рубрики
    for j in range(rubricsCount):
        if rubricName == foldersNames_list[j]:
            rubricNumber = j
    rubricsNumbers_list.append(rubricNumber)

# Таблица [Номер_рубрикик, Рубрика, Файл_текста, Текст_подготовленный]
df_preparedTexts = pd.DataFrame({'RubricNumber' : rubricsNumbers_list,
                                 'Rubric' : foldersPaths_all_rubrics,
                                 'FileName' : articlesTexts_prepared_filesNames,
                                 'TextPrep' : articlesTexts_prepared_list})




#
# Разделение всех статей на выборки Обучающую и Тестовую
#



# Представление текстов в виде числовых векторов
numericVectors_mx = 0 # Тексты в виде числоых векторов. Матрица размером: [Число_документов х Число_уникальных_слов_всего]
docs = articlesTexts_prepared_list #    Документы (тексты)
vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5, stop_words = None)
res = vectorizer.fit_transform(docs)
res_words = vectorizer.get_feature_names()  # Список слов из всех текстов
res_mx = res.todense()                     # Тексты в виде числоых векторов. Матрица размером: [Число_документов х Число_уникальных_слов_всего]
res_idf = vectorizer.idf_

numericVectors_mx = res_mx

# Сохранение числовых векторов в таблицу df_preparedTexts
df_preparedTexts['NumericVector'] = list(numericVectors_mx)


# Обучение модели KNN
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris


texts_dataset_data = numericVectors_mx                # Тексты в виде числовых векторов (X)
texts_dataset_target = np.array(rubricsNumbers_list)  # Метки текстов (номер рубрики для текста) (Y)

'''
# Для провекри на наборе данных iris_dataset
iris_dataset = load_iris()
iris_dataset_data = iris_dataset['data']
iris_dataset_target = iris_dataset['target']
'''

X_train, X_test, Y_train, Y_test = train_test_split(texts_dataset_data, texts_dataset_target)

model_KNN = KNeighborsClassifier(n_neighbors = 4) # Число соседей
model_KNN.fit(X_train, Y_train)


predicted = model_KNN.predict(X_test)

print(metrics.classification_report(Y_test, predicted))
print(metrics.confusion_matrix(Y_test, predicted))



# Определение текста, который соответствует числовому вектору из тестовой выборки
idx_text = 0                # Индекс текста в таблице
for i in range(articlesCount):
    # Если вектор из таблицы df_preparedTexts равен вектору из тестовой выборки X_test, то сохранение индекса
    if (df_preparedTexts['NumericVector'][i] == X_test[0]).all():
        idx_text = i        # Индекс нужного текста в таблице 
text_text = df_preparedTexts['TextPrep'][idx_text]
        


# Применение модели для первого текста тестовой выборки
predicted_text1 = model_KNN.predict(X_test[0])
print('\nПервый текст из тестовой выборки следующий:')
print(text_text)

print('\nПервый текст из тестовой выборки относится к следующей рубрике')
print('Номер рубрики: ', predicted_text1[0])
print('Название рубрики: ', foldersNames_list[predicted_text1[0]])

