from utils import *
from getKeyword import *

def main():

    tracking_file = open('track.txt', 'a')
    handle = ScrapeData("./sample.xlsx")
    urlList = handle.readFromExcel()
    handle.scrapy(urlList)
    execute()


if __name__ == '__main__':
  
    main()
