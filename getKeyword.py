import pandas as pd
import pandas as pd
import os
from gensim.parsing.preprocessing import strip_non_alphanum,strip_numeric,strip_punctuation

def readFromDir(osList):
    """
    This reads the scraped raw data
    """

    textList = []


    for i in range(len(osList)):
        filesList = []
        textArray = []
        for (dirpath, dirnames, filenames) in os.walk(osList[i]):
            filesList.extend(filenames)
            os.chdir(osList[i])
            for _ in range(len(filesList[:10])):
                with open('{}'.format(filesList[_]), 'r', encoding='utf-8') as file:
                    p = file.read()
                    textArray.append(p)

            text_arr = ','.join(textArray)
            text_arr = strip_punctuation(text_arr)
            text_arr = strip_numeric(text_arr)
            text_arr = strip_non_alphanum(text_arr)
            textList.append(text_arr)

        os.chdir('..')

    return textList


def execute():
    """
    main execution
    """
    df = pd.read_csv('./results.csv')
    osList1 = list(df['Raw_content'])

    osList = []
    for i in osList1:
        print(i)
        osList.append('./' + str(i))

    text = readFromDir(osList)
    df1 = pd.read_excel('./sample.xlsx')
    df['scraped_text'] = text
    df['class'] = df1['target_groups']
    df = df.applymap(lambda x:x.encode('unicode_escape').decode('utf-8') if isinstance(x,str) else x)
    df.to_excel('./m.xlsx')

if __name__ == "__main__":
    execute()