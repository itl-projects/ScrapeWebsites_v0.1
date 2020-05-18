import os
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import regex as re
from urllib.parse import urlparse
import os

class ScrapeData:

    def __init__(self, query):
        self.query = query

    def readFromExcel(self):
        """
        Reading from excel file
        """
        dataFrame = pd.read_excel(self.query)
        scrapyUrlList = list(dataFrame['web'][304:])
        return scrapyUrlList

    def appendToCsv(self, df, csvFilePath, sep=","):
        """
        Append results to csv file
        """
        if not os.path.isfile(csvFilePath):
            df.to_csv(csvFilePath, mode='a', index=False, sep=sep)

        elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):

            raise Exception(
                "Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(
                    len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")

        elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():

            raise Exception("Columns and column order of dataframe and csv file do not match!!")

        else:
            df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


    def scrapy(self, scrapyUrlList):
        """
        Scraping function for website in excel
        """
        anchorList = []

        for urls in range(len(scrapyUrlList)):
            with open('track.txt', 'r', encoding='UTF-8') as tracking_file:

                if scrapyUrlList[urls] not in tracking_file.read():

                    telNumber, mailId, facebookId, instagramId, otherLink = {}, {}, {}, {}, {}

                    anchorList, instagramList, facebookList, mailList, telList, osList, otherList = [], [], [], [], [], [], []
                    count = 0
                    url = scrapyUrlList[urls]
                    print(url)
                    try:
                        headers = {
                            "Accept-Language": "en",
                            "Accept": "text/html",
                        }
                        page = requests.get(url, headers=headers)
                        print(page.headers)
                        try:
                            soup = BeautifulSoup(page.content, 'html.parser')
                            for a in soup.find_all('a', href=True):
                                temp = a['href']
                                anchorList.append(temp)

                            for script in soup(["script", "style"]):
                                script.decompose()

                            text = soup.get_text()
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = ' '.join(chunk for chunk in chunks if chunk)

                        except:
                            anchorList.append('')
                            text = ''

                    except:
                        print("Error: Could not establish a connection")
                        text = ''
                        anchorList.append('')

                    try:
                        translateList = []
                        """
                        Translating the text from other language to english
                        """
                        blob = TextBlob(text)
                        englishString = blob.translate(to='en')
                        translateList.append(str(englishString))

                    except:
                        translateList.append(text)

                    url = scrapyUrlList[urls]
                    cleanAnchorList = []

                    uniqueAnchors = np.array(anchorList)
                    np.unique(uniqueAnchors)


                    instagramList = list(set(filter(lambda word: 'instagram' in word, uniqueAnchors)))
                    if instagramList == []:
                        instagramList.extend(re.findall('.+www.instagram.com\/[^\/]+\S+', text))
                    instagramList = ' '.join(instagramList[:2])

                    facebookList = list(set(filter(lambda word: 'facebook' in word, uniqueAnchors)))
                    if facebookList == []:
                        facebookList.extend(re.findall('.+www.facebook.com\/[^\/]+\S+', text))
                    facebookList = ' '.join(facebookList[:2])

                    telList = list(set(filter(lambda word: word.startswith('tel'), uniqueAnchors)))
                    if telList == []:
                        telList.extend(re.findall('[(][+][(]?[0-9]{1,3}[)]|[+]\d[\d\- ]{7,}\d', text))
                    res = [sub.replace('tel:', ' ') for sub in telList]
                    telList = ' '.join(res[:3])

                    mailList = list(set(filter(lambda word: word.startswith('mail'), uniqueAnchors)))
                    if mailList == []:
                        mailList.extend(re.findall('\S+@\S+', text))
                    res = [sub.replace('mailto:', ' ') for sub in mailList]
                    mailList = ' '.join(res[:3])

                    otherList.extend(list(set(filter(lambda word: 'maps.google' in word, uniqueAnchors))))
                    otherList.extend(list(set(filter(lambda word: 'wordpress.com' in word, uniqueAnchors))))
                    otherList.extend(list(set(filter(lambda word: 'linkedin.com' in word, uniqueAnchors))))
                    otherList.extend(list(set(filter(lambda word: 'twitter.com' in word, uniqueAnchors))))
                    otherList.extend(list(set(filter(lambda word: 'youtube.com' in word, uniqueAnchors))))

                    if otherList == []:
                        otherList.extend(re.findall('.+www.youtube.com\/[^\/]+\S+', text))
                        otherList.extend(re.findall('.+www.wordpress.com\/[^\/]+\S+', text))
                        otherList.extend(re.findall('.+www.linkedin.com\/[^\/]+\S+', text))
                        otherList.extend(re.findall('.+maps.google\S+', text))
                        otherList.extend(re.findall('.+www.twitter.com\/[^\/]+\S+', text))

                    otherList = ' '.join(otherList[:3])
                    if not url.endswith('/'):
                        url = url + '/'

                    try:
                        urlSplitList = url.split('//')[1].split('/')[0]
                        print(urlSplitList)
                        if os.path.exists("./{}".format(urlSplitList)):
                            if os.path.exists("./{}_part2".format(urlSplitList)):
                                if os.path.exists("./{}_part3".format(urlSplitList)):
                                    if os.path.exists("./{}_part4".format(urlSplitList)):
                                        if os.path.exists("./{}_part5".format(urlSplitList)):
                                            os.mkdir("./{}_part6".format(urlSplitList))
                                            os.chdir("./{}_part6".format(urlSplitList))
                                        else:
                                            os.mkdir("./{}_part5".format(urlSplitList))
                                            os.chdir("./{}_part5".format(urlSplitList))
                                    else:
                                        os.mkdir("./{}_part4".format(urlSplitList))
                                        os.chdir("./{}_part4".format(urlSplitList))
                                else:
                                    os.mkdir("./{}_part3".format(urlSplitList))
                                    os.chdir("./{}_part3".format(urlSplitList))
                            else:
                                os.mkdir("./{}_part2".format(urlSplitList))
                                os.chdir("./{}_part2".format(urlSplitList))

                        else:
                            os.mkdir("./{}".format(urlSplitList))
                            os.chdir("./{}".format(urlSplitList))
                    except:
                        print('path error')


                    osList.append(os.getcwd().split('\\')[-1])
                    print(osList)

                    try:
                        with open('{}.txt'.format(count), 'w', encoding='UTF-8') as makefile:
                            makefile.write("".join(translateList))

                            for lists in translateList:
                                print('Word count = {}'.format(len(lists.split())))
                    except:
                        print('Could not get data')

                    try:
                        base_url1 = url.split('.')[1]
                    except:
                        base_url1 = url
                    try:
                        base_url2 = url.split('//')[1].split('/')[0]
                    except:
                        base_url2 = url

                    removeList = ['facebook', 'twitter', 'youtube', 'instagram', 'linkedin', 'wordpress', 'maps', 'jpg',
                                  'png', 'pdf', 'mp3', 'mp4', 'google']


                    for steps in uniqueAnchors:

                        if any(c in str(steps) for c in removeList):
                            pass

                        elif str(steps).startswith('http') or str(steps).startswith('https'):

                            if 'www' in url:
                                if base_url1 in str(steps):
                                    cleanAnchorList.append(str(steps))
                            elif 'www' not in url:
                                if base_url2 in str(steps):
                                    cleanAnchorList.append(str(steps))


                        elif str(steps).endswith('jpg') or str(steps).startswith('javascript') or str(steps).startswith(
                                'tel') or str(steps).startswith('mail'):
                            pass

                        elif str(steps).startswith('/'):

                            if url.endswith('/'):
                                url = url[:-1]
                            finalUrl = url + steps
                            cleanAnchorList.append(finalUrl)

                        else:
                            if url.endswith('/'):
                                finalUrl = url + steps
                                cleanAnchorList.append(finalUrl)
                            else:
                                finalUrl = url + '/' + steps
                                cleanAnchorList.append(finalUrl)

                    cleanAnchorList = list(set(cleanAnchorList))
                    print('Number of urls to be scraped from base url = {}, {}'.format(len(cleanAnchorList),
                                                                                       cleanAnchorList))
                    instagramId[url] = instagramList
                    facebookId[url] = facebookList
                    mailId[url] = mailList
                    telNumber[url] = telList
                    otherLink[url] = otherList
                    count1 = 0

                    base = urlparse(url)
                    one = base.scheme + '://' + base.netloc
                    two = 'https://' + base.netloc
                    try:
                        three = 'http://' + base.netloc.split('.')[0] + '.' + base.netloc.split('.')[2] + '.' + \
                                base.netloc.split('.')[-1]
                        cleanAnchorList.append(three)
                        print(three)
                        try:
                            four = 'http://' + base.netloc.split('.')[0] + '.' + base.netloc.split('.')[1] + '.' + \
                                   base.netloc.split('.')[-1]
                            cleanAnchorList.append(four)
                            print(four)
                        except:
                            cleanAnchorList.append('')
                    except:
                        cleanAnchorList.append('')

                    cleanAnchorList.append(one)
                    cleanAnchorList.append(two)
                    print(cleanAnchorList)

                    for i in cleanAnchorList[:50]:
                        print(i)

                        try:
                            headers = {
                                "Accept-Language": "en",
                                "Accept": "text/html",
                            }
                            page = requests.get(i, headers = headers)
                            html_code = page.content

                            try:
                                soup = BeautifulSoup(html_code, 'html.parser')
                                for script in soup(["script", "style"]):
                                    script.decompose()

                                text = soup.get_text()
                                lines = (line.strip() for line in text.splitlines())
                                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                                text = ' '.join(chunk for chunk in chunks if chunk)
                            except:
                                anchorList.append('')
                                text = ''

                        except:
                            print("Error: Could not establish a connection")
                            text = ''
                            anchorList.append('')

                        try:
                            englishList = []
                            blob = TextBlob(text)
                            englishLine = blob.translate(to='en')
                            englishList.append(str(englishLine))
                            count1 = count1 + 1

                        except:
                            englishList.append(text)
                            count1 = count1 + 1

                        try:
                            with open('{}.txt'.format(count1), 'w', encoding='UTF-8') as file:
                                file.write("".join(englishList))

                                for li in englishList:
                                    print('Word count = {}'.format(len(li.split())))

                        except:
                            print('Could not get data')
                    print("Finished . .")

                    os.chdir("..")

                    dataframe = pd.DataFrame(list(
                        zip(telNumber.keys(), telNumber.values(), mailId.values(), instagramId.values(),
                            facebookId.values(),
                            otherLink.values(), osList)),
                                             columns=['Website', 'Phone Number', 'Email', 'Instagram', 'Facebook',
                                                      'Others', 'Raw_content'])
                    print(dataframe)

                    ScrapeData.appendToCsv(self, df=dataframe, csvFilePath='./results.csv', sep=",")

                    with open('track.txt', 'a', encoding='UTF-8') as tracking_file:
                        tracking_file.write(scrapyUrlList[urls])
                        tracking_file.write("\n")
