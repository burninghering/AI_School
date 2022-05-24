from selenium import webdriver
import time
from urllib.request import (urlopen,urlparse,urlretrieve)

from selenium.webdriver.common.keys import Keys
import pandas as pd

import os

# 구글 이미지 URL
chrome_path="./chromedriver.exe"

base_url="https://chrome.google.co.kr/imghp"

# 구글 검색할 때는 옵션이 필요함
chrome_options=webdriver.ChromeOptions()
chrome_options.add_argument("lang=ko_KR") #한국어
# 윈도우 창 사이즈 바꾸기
chrome_options.add_argument("window-size=1920x1080")

# 드라이버 연결
driver = webdriver.Chrome(chrome_path,chrome_options=chrome_options)
driver.get(base_url)
# 엘리멘트 로드될 때까지 지정한 시간만큼 대기할 수 있도록 하는 옵션
driver.implicitly_wait(3)
# 잘 나오는지 캡쳐
driver.get_screenshot_as_file("google_screen.png")
# 드라이버 종료
driver.close()

# 스크롤 조절
def selenium_scroll_option():
    SCROLL_PAUSE_SEC = 1

    # 스크롤 높이 가져옴
    last_height=driver.execute_script(
        "return document.body.scrollHeight") #구글에서 키워드로 가져오면 된다
    
    while True:
        # 끝까지 스크롤 다운
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
        # 대기 타임 걸기
        time.sleep(SCROLL_PAUSE_SEC)
        # 스크롤 다운 후 스크롤 높이 다시 가져오기
        new_height=driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break
        last_height=new_height

a='공룡'
image_name="dinosaur"
driver=webdriver.Chrome(chrome_path) #드라이버 불러오기
driver.get("http://www.google.co.kr/imghp?hl=ko") #어디 url에 검색할 지 인자 넣어줌
browser=driver.find_element_by_name('q') #검색어 키
browser.send_keys(a)
browser.send_keys(Keys.RETURN)



#사진 더 보기
#//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input
selenium_scroll_option #스크롤하여 이미지 확보
driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[2]/div[2]/input').click() #버튼 누르기
selenium_scroll_option() #스크롤 내리기

#이미지 저장 src 요소를 리스트업 해서 이미지 url 저장
image=driver.find_elements_by_css_selector(".rg_i.Q4Luwd")
#클래스 네임에서 공백은 .을 찍어줌

#이미지 가져오기
image_url=[]
for i in image:
    if i.get_attribute("src")!=None:
        image_url.append(i.get_attribute("src"))
    else:
        image_url.append(i.get_attribute("data-src"))

#전체 이미지 개수 살펴보기
print(f"전체 다운로드한 이미지 개수 : {len(image_url)}")

#판다스에 한 차원 내려서 저장하기
image_url=pd.DataFrame(image_url)[0].unique()

#해당하는 파일에 이미지를 다운로드
#dino 디렉토리를 만들기
os.makedirs("./dino",exist_ok=True)
dino="./dino/"

if image_name=='shark':
    for t,url in enumerate(image_url,0):
        urlretrieve(url,dino+image_name+"_"+str(t)+".png") #다운해주는 기능

    
    driver.close()

print("완료")