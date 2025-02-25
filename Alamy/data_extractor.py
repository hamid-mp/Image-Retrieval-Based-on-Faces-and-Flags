from bs4 import BeautifulSoup
import requests






class ExractInformation():

    def __init__(self, webpage):
        self.soup = BeautifulSoup(webpage, 'html.parser')

        self.num_pages = self.soup.find_all('div', attrs={'data-testid':'page-number'})[0].get_text()
        self.num_pages = int(self.num_pages.split(' ')[-1])

    @staticmethod
    def page_links(new_webpage):
        new_webpage = BeautifulSoup(new_webpage, 'html.parser')
        links = new_webpage.find_all('a', attrs={'class':'z-2 absolute top-0 h-full bg-black bg-opacity-20 opacity-0 duration-200 group-hover:opacity-100'})
        hrefs = ['https://www.alamy.com'+link.get('href') for link in links if link.get('href') is not None]
        return hrefs
    

    @staticmethod
    def image_downloader(webpage):
        webpage = BeautifulSoup(webpage, 'html.parser')
        image_url = webpage.find('image')['href']
        

        img_data = requests.get(image_url).content
        return img_data
