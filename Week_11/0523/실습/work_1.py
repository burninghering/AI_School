from selenium import webdriver
from bs4 import BeautifulSoup as bs
import time
import pandas as pd

search_keyword = ["감자칩", "멜론", "수박"]

driver = webdriver.Chrome("./chromedriver.exe")
driver.implicitly_wait(3)
time.sleep(5)

def search_auction(keyword):
    driver.get("http://www.auction.co.kr/")
    time.sleep(5)

    driver.find_element_by_xpath('//*[@id="txtKeyword"]').send_keys(keyword)
    time.sleep(2)

    driver.find_element_by_xpath('//*[@id="core_header"]/div/div[1]/form/div[1]/input[2]').click()

    html = driver.page_source
    soup = bs(html, 'html.parser')
    itemlist = soup.findAll('div', {"class": "section--itemcard"})
    time.sleep(5)

    titles = []
    prices = []
    links = []
    for item in itemlist:
        title = item.find("span", {"class": "text--title"}).text
        titles.append(title)
        price = item.find("strong", {"class": "text--price_seller"}).text
        prices.append(price)
        link = item.find("span", {"class": "text--itemcard_title ellipsis"}).a['href']
        links.append(link)

    time.sleep(3)

    df = pd.DataFrame({'상품명': titles, '가격': prices, '상품 링크': links})
    df.to_csv(f'./auction_{keyword}.csv', index=False, encoding='utf-8-sig')

for key in search_keyword :
    print(key)
    search_auction(key)

driver.close()