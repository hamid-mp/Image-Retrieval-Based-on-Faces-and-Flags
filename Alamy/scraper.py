import requests
import base64
import time
import configparser
from pathlib import Path
import psycopg2
import re
import os
from bs4 import BeautifulSoup
#from requests_html import HTMLSession
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from data_extractor import ExractInformation
import argparse

Path('.\\download_links').mkdir(exist_ok=True, parents=True)



def read_country_name(pth):
    names = []
    with open(pth, 'r') as f:
        lines = f.readlines()
        for l in lines:
            names.append(l.strip())
    return names


class AlamyScraper():
    def __init__(self):

        self.driver = None

    def load_driver(self):
        chrome_options = Options()
        chrome_options.add_argument('--disable-notifications')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)




    def process(self):
        for country in COUNTRIES_NAME:
            c = 0
            country = country.lower()
            Path(f'./download_links/{country}').mkdir(exist_ok=True, parents=True)
            url = f'https://www.alamy.com/stock-photo/{country}-flag.html?page=1&sortBy=relevant'
            self.driver.get(url)
            time.sleep(15)
            try:
                self.driver.find_element("xpath",'//button[@data-tid="banner-accept"]').click()
            except:
                pass
            webpage = self.driver.page_source
            webpage_handler = ExractInformation(webpage)
            num_pages = webpage_handler.num_pages
            num_pages = 60 if num_pages > 60 else num_pages

            for i in range(2, num_pages):
                
                try:
                    hrefs = webpage_handler.page_links(webpage)
                    for href in hrefs:
                        try:
                            self.driver.find_element("xpath",'//button[@data-tid="banner-accept"]').click()
                        except:
                            pass
                        c += 1
                        self.driver.get(href)
                        try:
                            self.driver.find_element("xpath",'//button[@title="Enlarge"]').click()
                        except:
                            pass
                        webpage = self.driver.page_source
                        
                        img_data = webpage_handler.image_downloader(webpage)
                        with open(f'./download_links/{country}/{c:05d}.jpg', 'wb') as handler:
                            handler.write(img_data)

                    if c>1500:
                        break

                    url = f'https://www.alamy.com/stock-photo/{country}-flag.html?page={i}&sortBy=relevant'

                    self.driver.get(url)
                    time.sleep(10)
                    try:
                        self.driver.find_element("xpath",'//button[@data-tid="banner-accept"]').click()
                    except:
                        pass
                    webpage = self.driver.page_source

                except:
                    continue




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--names', default='Path/to/.txt', type=str, help="Path to txt file contains country names")
    args = parser.parse_args()
    COUNTRIES_NAME = read_country_name(args.names)

    scraper = AlamyScraper()
    scraper.load_driver()


    scraper.process()            
