
# 연도별로 작업해서 합칠게요.
# 컬럼명은 ['title', 'reviews']로 통일하겠습니다.
# 파일명은 reviews_{연도}.csv 로 하겠습니다.
# 경민님이 2013-2015
# 재희님이 2016-2023 해주세요.


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time
import datetime

options = ChromeOptions()
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36"
options.add_argument('user-agent=' + user_agent)
options.add_argument("lang=ko_KR")
# options.add_argument('headless')
# options.add_argument('window-size=1920x1080')
# options.add_argument("disable-gpu")
# options.add_argument("--no-sandbox")

service = ChromeService(executable_path=ChromeDriverManager().install())

driver = webdriver.Chrome(service=service, options=options)  # <- options로 변경

# year = 2023
# url = 'https://movie.daum.net/ranking/boxoffice/monthly?date={}01'.format(year)
title_path = '//*[@id="mainContent"]/div/div[2]/ol/li[{}]/div/div[2]/strong/a'
button_score_path = '//*[@id="mainContent"]/div/div[2]/div[1]/ul/li[4]/a/span'
button_more_path = '//*[@id="alex-area"]/div/div/div/div[3]/div[1]/button'
review_path = '/html/body/div[2]/main/article/div/div[2]/div[2]/div/div/div[2]/div/div/div/div[3]/ul[2]/li[{}]/div/p'
button_next_path = '//*[@id="mainContent"]/div/div[1]/div[1]/div/a[2]/span'
url = 'https://movie.daum.net/ranking/boxoffice/monthly?date={}01'

for year in range(2013, 2024):
    driver.get(url.format(year))
    time.sleep(2)
    crawled_titles = []
    for k in range(1, 13):
        for i in range(1, 31):
            title = driver.find_element(By.XPATH, title_path.format(i)).text
            if title in crawled_titles:
                continue
            crawled_titles.append(title)
            driver.find_element(By.XPATH, title_path.format(i)).click()
            time.sleep(0.5)
            driver.find_element(By.XPATH, button_score_path).click()
            time.sleep(1)
            for _ in range(5):
                try:
                    driver.find_element(By.XPATH, button_more_path).click()
                    time.sleep(1)
                except:
                    print('more', '{} {}'.format(i, _))
            reviews = []
            titles = []
            for j in range(1, 161):
                try:
                    review = driver.find_element(By.XPATH, review_path.format(j)).text
                    titles.append(title)
                    reviews.append(review)
                except:
                    print('review', '{} {}'.format(i, j))
            df = pd.DataFrame({'title':titles, 'reviews':reviews})
            try:
                df.to_csv('./crawling_data/reviews_{}_{}_{}.csv'.format(year, k, i), index=False)
            except:
                print(title)
            print(len(titles), len(reviews))
            print(titles)
            print(reviews)
            driver.back()
            time.sleep(0.5)
            driver.back()
            time.sleep(0.5)
        driver.find_element(By.XPATH, button_next_path).click()
        print(len(crawled_titles))
        print(crawled_titles)













